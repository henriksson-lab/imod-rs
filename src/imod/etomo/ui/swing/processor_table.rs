//! `IMOD/Etomo/src/etomo/ui/swing/ProcessorTable.java`.
//!
//! The native Swing cells, load-monitor processes, and the CPU/queue/GPU table
//! subclasses are their own source units.  This unit retains the complete common
//! table state and its row/list protocol instead of creating a replacement UI.
#![allow(dead_code)]

use std::collections::{BTreeMap, BTreeSet};

pub use super::processor_table_row::ProcessorTableRow;

pub const RUNNABLE_KEY: &str = "ProcessorTable";
pub const RESOURCE_KEY: &str = "ProcessorResourceTable";
pub const CPU_TYPE_LABEL: &str = "CPU Type";
pub const RESTARTS_LABEL: &str = "Retries";
pub const NUMBER_CPUS_MAX_LABEL: &str = "Max.";
pub const FIRST_QUEUE_LABEL: &str = "1st";
pub const SECONDARY_QUEUE_LABEL: &str = "2nd";
pub const NUMBER_CPUS_USED_LABEL2: &str = "Used";

/// Java private static inner `ColumnName` identity singletons.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ColumnName {
    NumberUsed,
    Number,
    Load,
    Type,
    Speed,
    Memory,
    Os,
    Run,
    Users,
}
impl std::fmt::Display for ColumnName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::NumberUsed => "NUMBER_USED",
            Self::Number => "NUMBER",
            Self::Load => "LOAD",
            Self::Type => "TYPE",
            Self::Speed => "SPEED",
            Self::Memory => "MEMORY",
            Self::Os => "OS",
            Self::Run => "RUN",
            Self::Users => "USERS",
        })
    }
}

/// `etomo.ui.QueueTableEvent`, consumed by ProcessorTable and its rows.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QueueTableEvent {
    AllowDisplay,
    Display,
    Displayed,
    DisableSecondaryQueue,
    EnableSecondaryQueue,
    Hidden,
    PreventDisplay,
}

/// Java private final inner `RowList`.
#[derive(Clone, Debug, Default)]
pub struct RowList {
    pub list: Vec<ProcessorTableRow>,
    pub contracted_index: Vec<usize>,
}
impl RowList {
    pub fn queue_table_event_action(&mut self, event: QueueTableEvent) {
        for row in &mut self.list {
            row.queue_table_event_action(event);
        }
    }
    pub fn secondary_queue_selected_action(&mut self) {
        for row in &mut self.list {
            row.secondary_queue_selected_action();
        }
    }
    pub fn set_computer_map(&mut self, map: &BTreeMap<String, String>) {
        if map.is_empty() {
            return;
        }
        for row in &mut self.list {
            row.set_selected(false);
            if let Some(cpus) = map.get(&row.computer) {
                row.set_selected(true);
                row.set_cpus_selected(cpus);
            }
        }
    }
    pub fn set_secondary_queue(&mut self, name: Option<&str>, queue_table: bool) {
        let Some(name) = name else { return };
        if !queue_table {
            return;
        }
        for row in &mut self.list {
            if row.has_secondary_queue && row.computer == name {
                row.set_secondary_queue_selected();
                return;
            }
        }
    }
    pub fn set_selected_error(&mut self) {
        for row in &mut self.list {
            row.set_selected_error();
        }
    }
    pub fn add(&mut self, row: ProcessorTableRow) {
        self.list.push(row);
    }
    pub fn display(&mut self, expanded: bool) {
        let indexes = if expanded {
            (0..self.list.len()).collect()
        } else {
            self.contracted_index.clone()
        };
        for (display_index, index) in indexes.into_iter().enumerate() {
            self.list[index].delete_row();
            self.list[index].display(display_index);
        }
    }
    pub fn size(&self, expanded: bool) -> usize {
        if expanded {
            self.list.len()
        } else {
            self.contracted_index.len()
        }
    }
    pub fn get(&self, index: isize) -> Option<&ProcessorTableRow> {
        usize::try_from(index).ok().and_then(|i| self.list.get(i))
    }
    pub fn get_mut(&mut self, index: isize) -> Option<&mut ProcessorTableRow> {
        usize::try_from(index)
            .ok()
            .and_then(|i| self.list.get_mut(i))
    }
    pub fn get_by_computer(&self, computer: &str) -> Option<&ProcessorTableRow> {
        self.list.iter().find(|row| row.computer == computer)
    }
    pub fn get_by_computer_mut(&mut self, computer: &str) -> Option<&mut ProcessorTableRow> {
        self.list.iter_mut().find(|row| row.computer == computer)
    }
    pub fn set_contracted_index(&mut self, expanded: bool) {
        self.contracted_index.clear();
        if !expanded {
            self.contracted_index
                .extend(self.list.iter().enumerate().filter_map(|(i, row)| {
                    (row.selected || row.secondary_queue_selected).then_some(i)
                }));
        }
    }
    pub fn get_parameters(&self, param: &mut BTreeMap<String, String>, secondary: bool) {
        for row in &self.list {
            row.get_parameters(param, secondary);
        }
    }
    pub fn set_selected(&mut self, index: isize, selected: bool) {
        if let Some(row) = self.get_mut(index) {
            row.set_selected(selected);
        }
    }
    pub fn enable_selection_field(&mut self, name: &str, enabled: bool) {
        if let Some(row) = self.get_by_computer_mut(name) {
            row.enable_selection_field(enabled);
        }
    }
    pub fn reset_results(&mut self) {
        for row in &mut self.list {
            row.reset_results();
        }
    }
    pub fn get_total_successes(&self) -> i32 {
        self.list.iter().map(|row| row.successes).sum()
    }
    pub fn get_cpus_selected(&self) -> i32 {
        self.list.iter().map(|row| row.cpus_selected).sum()
    }
    pub fn get_first_selected_index(&self) -> isize {
        self.list
            .iter()
            .position(|row| row.selected)
            .map_or(-1, |i| i as isize)
    }
    pub fn get_first_selected_secondary_queue_index(&self) -> isize {
        self.list
            .iter()
            .position(|row| row.secondary_queue_selected)
            .map_or(-1, |i| i as isize)
    }
    pub fn get_selected_secondary_queue(&self) -> Option<&ProcessorTableRow> {
        self.list.iter().find(|row| row.secondary_queue_selected)
    }
    pub fn get_next_selected_index(&self, last: isize) -> isize {
        self.list
            .iter()
            .enumerate()
            .skip((last + 1).max(0) as usize)
            .find_map(|(i, row)| row.selected.then_some(i as isize))
            .unwrap_or(-1)
    }
    pub fn get_computer(&self, index: isize) -> Option<&str> {
        self.get(index).map(|row| row.computer.as_str())
    }
    pub fn clear_failure_reason(&mut self, selected: bool) {
        for row in &mut self.list {
            if !selected || row.selected {
                row.clear_failure_reason(None);
            }
        }
    }
    pub fn store(&self, props: &mut BTreeMap<String, String>, prepend: &str) {
        for row in &self.list {
            row.store(props, prepend);
        }
    }
    pub fn load(&mut self, props: &BTreeMap<String, String>, prepend: &str) {
        for row in &mut self.list {
            row.load(props, prepend);
        }
    }
}

/// The abstract methods in Java `ProcessorTable`, supplied by Cpu/Gpu/Queue tables.
pub trait ProcessorTableHooks {
    /// Java abstract `getSize`.
    fn get_size(&self) -> usize;
    /// Java abstract `getNode(int)`. `Node.java` is the direct storage boundary;
    /// this returned name is the exact value ProcessorTable consumes from it.
    fn get_node(&self, index: usize) -> Option<String>;
    /// Java abstract `createProcessorTableRow(ProcessorTable, Node, int,
    /// ProcessorTableState)`.
    fn create_processor_table_row(&self, node: &str, num_rows_in_table: usize)
    -> ProcessorTableRow;
    fn get_header1_computer_text(&self) -> String;
    /// Java abstract `getIntermittentCommand(String)`; the command class is a
    /// direct process boundary, while this string remains its computer key.
    fn get_intermittent_command(&self, computer: &str) -> Option<String>;
    /// Java abstract `isExcludeNode(Node)`.
    fn is_exclude_node(&self, node: &str) -> bool;
    /// Java abstract `isNiceable`.
    fn is_niceable(&self) -> bool;
    fn get_store_prepend(&self) -> String;
    fn get_load_prepend(&self, version: &str) -> String;
    /// Java abstract `initRow(ProcessorTableRow)`.
    fn init_row(&self, row: &mut ProcessorTableRow);
    fn is_queue_table(&self) -> bool;
    fn is_cpu_table(&self) -> bool;
    fn is_gpu_table(&self) -> bool;
    fn get_no_cpus_selected_error_message(&self) -> String;
}

/// Java `ProcessorTable`.  `displayed_fields` represents `ProcessorTableState.isUse`.
pub struct ProcessorTable<H: ProcessorTableHooks> {
    pub hooks: H,
    pub row_list: RowList,
    pub header1_computer: String,
    pub header2_computer: String,
    pub header1_number_cpus: String,
    pub header2_number_cpus_used: String,
    pub displayed_fields: BTreeSet<String>,
    pub header_rows: [Vec<String>; 2],
    pub table_visible: bool,
    pub expanded: bool,
    pub stopped: bool,
    pub secondary: bool,
    pub limited: bool,
    pub runnable: bool,
    pub scrolling: bool,
    pub load_monitor: bool,
    pub load_monitor_restarts: u32,
    pub repaint_count: u32,
    pub pack_count: u32,
    pub selection_changed_count: u32,
    pub process_events: Vec<&'static str>,
    pub queue_table_events: Vec<QueueTableEvent>,
}
impl<H: ProcessorTableHooks> ProcessorTable<H> {
    /// Java constructor; `no_load` is `EtomoDirector.INSTANCE.getArguments().isNoLoad()`.
    pub fn new(
        hooks: H,
        displayed_fields: BTreeSet<String>,
        _display_queues: bool,
        runnable: bool,
        no_load: bool,
    ) -> Self {
        let header1_computer = hooks.get_header1_computer_text();
        Self {
            hooks,
            row_list: RowList::default(),
            header1_computer,
            header2_computer: String::new(),
            header1_number_cpus: "# Cores".into(),
            header2_number_cpus_used: NUMBER_CPUS_USED_LABEL2.into(),
            displayed_fields,
            header_rows: [vec![], vec![]],
            table_visible: true,
            expanded: false,
            stopped: true,
            secondary: false,
            limited: false,
            runnable,
            scrolling: false,
            load_monitor: !no_load,
            load_monitor_restarts: 0,
            repaint_count: 0,
            pack_count: 0,
            selection_changed_count: 0,
            process_events: vec![],
            queue_table_events: vec![],
        }
    }
    pub fn get_focusable_parents(&self) -> [&str; 1] {
        ["rootPanel"]
    }
    pub fn getheader1_number_cpus_title(&self) -> &str {
        "# Cores"
    }
    pub fn set_header1_number_cpus_title(&mut self) {
        self.header1_number_cpus = self.getheader1_number_cpus_title().into();
    }
    pub fn set_header1_number_cpus_title_to(&mut self, title: impl Into<String>) {
        self.header1_number_cpus = title.into();
    }
    pub fn create_table(&mut self) {
        self.expanded = true;
        self.init_table();
        self.build();
    }
    pub fn is_valid(&self) -> bool {
        !self.row_list.list.is_empty()
    }
    pub fn set_expanded(&mut self, expanded: bool) {
        if self.expanded == expanded {
            return;
        }
        self.expanded = expanded;
        self.row_list.set_contracted_index(expanded);
        self.build();
    }
    pub fn set_visible(&mut self, visible: bool) {
        self.table_visible = visible;
    }
    pub fn build(&mut self) {
        self.build_table();
        self.repaint_count += 1;
    }
    pub fn init_table(&mut self) {
        // Java's node loop.  Node attribute/excluded-interface filtering belongs to
        // Node.java; `is_exclude_node` is the subclass part of the same predicate.
        let size = self.hooks.get_size();
        for index in 0..size {
            if let Some(node) = self.hooks.get_node(index) {
                if !self.hooks.is_exclude_node(&node) {
                    let mut row = self.hooks.create_processor_table_row(&node, size);
                    self.hooks.init_row(&mut row);
                    self.row_list.add(row);
                }
            }
        }
        if self.row_list.list.len() == 1 {
            self.row_list.set_selected(0, true);
        }
        self.set_tool_tip_text();
    }
    pub fn msg_viewport_paged(&mut self) {
        self.build();
        self.pack_count += 1;
    }
    pub fn set_secondary(&mut self, input: bool) {
        if input != self.secondary {
            self.secondary = input;
            self.build();
            self.row_list.set_selected_error();
            self.pack_count += 1;
        }
    }
    pub fn build_table(&mut self) {
        let use_field = |field: &str| self.displayed_fields.contains(field);
        let mut one = vec![
            self.header1_computer.clone(),
            self.header1_number_cpus.clone(),
        ];
        let mut two = vec![
            self.header2_computer.clone(),
            self.header2_number_cpus_used.clone(),
        ];
        if use_field("TWO_QUEUES_H2") {
            two.splice(
                0..0,
                [FIRST_QUEUE_LABEL.into(), SECONDARY_QUEUE_LABEL.into()],
            );
        }
        if use_field("NUM_CPUS_MAX_H2") {
            two.push(NUMBER_CPUS_MAX_LABEL.into());
        }
        for (field, top, bottom) in [
            ("NUM_GPUS", "GPUs", ""),
            ("LOAD_AVERAGE_H1", "Load Average", "1 Min."),
            ("CPU_USAGE_H1", "CPU Usage", ""),
            ("USERS_H1", "Users", ""),
            ("TYPE_H1", CPU_TYPE_LABEL, ""),
            ("SPEED_H1", "Speed", ""),
            ("MEMORY_H1", "RAM", ""),
            ("OS_H1", "OS", ""),
            ("GPU_TYPE_H1", "Type", ""),
            ("GPU_SPEED_H1", "Speed", ""),
            ("GPU_MEMORY_H1", "RAM", ""),
            ("GPU_NCORES_H1", "Cores", ""),
        ] {
            if use_field(field) {
                one.push(top.into());
                two.push(bottom.into());
            }
        }
        if use_field("LOAD_AVERAGE_H1") {
            two.push("5 Min.".into());
        }
        if use_field("RESTARTS_H1") {
            one.extend([RESTARTS_LABEL.into(), "Chunks".into(), "Failure".into()]);
            two.extend([String::new(), "Done".into(), "Reason".into()]);
        }
        self.header_rows = [one, two];
        self.row_list.display(self.expanded);
    }
    pub fn add(
        &mut self,
        cell: impl Into<String>,
        use_cell: bool,
        column_name: ColumnName,
        last_column_name: ColumnName,
    ) {
        if use_cell {
            let _remainder = column_name == last_column_name;
            self.header_rows[0].push(cell.into());
        }
    }
    pub fn get_container(&self) -> bool {
        self.table_visible
    }
    pub fn get_table_panel(&self) -> &[Vec<String>; 2] {
        &self.header_rows
    }
    pub fn get_table_layout(&self) -> &'static str {
        "GridBagLayout"
    }
    pub fn get_table_constraints(&self) -> &'static str {
        "GridBagConstraints"
    }
    pub fn reset_results(&mut self) {
        self.row_list.reset_results();
    }
    pub fn get_total_successes(&self) -> i32 {
        self.row_list.get_total_successes()
    }
    pub fn msg_cpus_selected_changed(&mut self) {
        self.selection_changed_count += 1;
    }
    pub fn msg_ending_process(&mut self) {
        self.process_events.push("msgEndingProcess");
    }
    pub fn msg_killing_process(&mut self) {
        self.process_events.push("msgKillingProcess");
    }
    pub fn msg_process_started(&mut self) {
        self.process_events.push("msgProcessStarted");
    }
    pub fn msg_pausing_process(&mut self) {
        self.process_events.push("msgPausingProcess");
    }
    pub fn get_cpus_selected(&self) -> i32 {
        self.row_list.get_cpus_selected()
    }
    pub fn restart_load_monitor(&mut self) {
        if self.load_monitor {
            self.load_monitor_restarts += 1;
        }
    }
    pub fn is_secondary(&self) -> bool {
        self.secondary
    }
    pub fn is_runnable(&self) -> bool {
        self.runnable
    }
    pub fn is_limited(&self) -> bool {
        self.limited
    }
    pub fn set_runnable(&mut self, input: bool) {
        if input != self.runnable {
            self.runnable = input;
            self.build();
            self.row_list.set_selected_error();
            self.pack_count += 1;
        }
    }
    pub fn set_limited(&mut self, input: bool) {
        if input != self.limited {
            self.limited = input;
            self.build();
            self.row_list.set_selected_error();
            self.pack_count += 1;
        }
    }
    pub fn get_first_selected_index(&self) -> isize {
        self.row_list.get_first_selected_index()
    }
    pub fn get_next_selected_index(&self, last: isize) -> isize {
        self.row_list.get_next_selected_index(last)
    }
    pub fn get_parameters(&self, parameter: &mut BTreeMap<String, String>) {
        self.row_list.get_parameters(parameter, self.secondary);
    }
    pub fn set_parameters(&mut self, parameter: &BTreeMap<String, String>) {
        self.set_computer_map(parameter);
    }
    pub fn get_machine_map(
        &self,
        parameter: &BTreeMap<String, String>,
    ) -> BTreeMap<String, String> {
        parameter.clone()
    }
    pub fn get_first_selected_computer(&self) -> Option<&str> {
        self.row_list
            .get_computer(self.row_list.get_first_selected_index())
    }
    pub fn get_selected_secondary_queue_node(&self) -> Option<&str> {
        self.row_list
            .get_selected_secondary_queue()
            .map(|row| row.computer.as_str())
    }
    pub fn size(&self) -> usize {
        self.row_list.size(self.expanded)
    }
    pub fn get_row(&self, computer: &str) -> Option<&ProcessorTableRow> {
        self.row_list.get_by_computer(computer)
    }
    pub fn get_first_selected_row(&self) -> Option<&ProcessorTableRow> {
        self.row_list.get(self.row_list.get_first_selected_index())
    }
    pub fn get_first_selected_secondary_queue_row(&self) -> Option<&ProcessorTableRow> {
        self.row_list
            .get(self.row_list.get_first_selected_secondary_queue_index())
    }
    pub fn add_restart(&mut self, computer: &str) {
        if let Some(row) = self.row_list.get_by_computer_mut(computer) {
            row.add_restart();
        }
    }
    pub fn add_success(&mut self, computer: &str) {
        if let Some(row) = self.row_list.get_by_computer_mut(computer) {
            row.add_success();
        }
    }
    pub fn set_computer_map(&mut self, map: &BTreeMap<String, String>) {
        self.row_list.set_computer_map(map);
        self.selection_changed_count += 1;
    }
    pub fn set_secondary_queue(&mut self, queue: Option<&str>) {
        self.row_list
            .set_secondary_queue(queue, self.hooks.is_queue_table());
        self.selection_changed_count += 1;
    }
    pub fn msg_dropped(&mut self, computer: &str, reason: impl Into<String>) {
        if let Some(row) = self.row_list.get_by_computer_mut(computer) {
            row.msg_dropped(reason);
        }
    }
    pub fn get_help_message(&self) -> String {
        format!(
            "Click on check boxes in the {} column and use the spinner in the {} {} column where available.",
            self.header1_computer, self.header1_number_cpus, self.header2_number_cpus_used
        )
    }
    pub fn start_load(&mut self) {
        if !self.load_monitor || self.secondary {
            return;
        }
        self.stopped = false;
    }
    pub fn get_intermittent_command(&self, index: isize) -> Option<String> {
        self.row_list
            .get_computer(index)
            .and_then(|computer| self.hooks.get_intermittent_command(computer))
    }
    pub fn end_load(&mut self) {
        if self.load_monitor {
            self.stopped = true;
        }
    }
    pub fn stop_load(&mut self) {
        if self.load_monitor {
            self.stopped = true;
        }
    }
    pub fn is_stopped(&self) -> bool {
        self.stopped
    }
    pub fn set_load(
        &mut self,
        computer: &str,
        load1: f64,
        load5: f64,
        users: i32,
        tooltip: impl Into<String>,
    ) {
        if let Some(row) = self.row_list.get_by_computer_mut(computer) {
            row.set_load(load1, load5, users, tooltip);
        }
    }
    pub fn set_load_array(&mut self, computer: &str, array: &[String]) {
        if let Some(row) = self.row_list.get_by_computer_mut(computer) {
            row.set_load_array(array);
        }
    }
    pub fn set_cpu_usage(&mut self, computer: &str, usage: f64, processors: impl Into<String>) {
        if let Some(row) = self.row_list.get_by_computer_mut(computer) {
            row.set_cpu_usage(usage, processors);
        }
    }
    pub fn msg_load_failed(
        &mut self,
        computer: &str,
        reason: impl Into<String>,
        tooltip: impl Into<String>,
    ) {
        if let Some(row) = self.row_list.get_by_computer_mut(computer) {
            row.clear_load(reason, tooltip);
        }
    }
    pub fn msg_starting_process_on_selected_computers(&mut self) {
        self.clear_failure_reason(true);
    }
    pub fn msg_starting_process(&mut self, computer: &str, first: &str, second: &str) {
        if let Some(row) = self.row_list.get_by_computer_mut(computer) {
            row.clear_failure_reason(Some((first, second)));
        }
    }
    pub fn clear_failure_reason(&mut self, selected: bool) {
        self.row_list.clear_failure_reason(selected);
    }
    pub fn get_group_key(&self) -> &'static str {
        if self.runnable {
            RUNNABLE_KEY
        } else {
            RESOURCE_KEY
        }
    }
    pub fn store(&self, props: &mut BTreeMap<String, String>) {
        self.store_with_prepend(props, "");
    }
    pub fn store_with_prepend(&self, props: &mut BTreeMap<String, String>, prepend: &str) {
        let part = self.hooks.get_store_prepend();
        let full = if prepend.is_empty() {
            part
        } else {
            format!("{prepend}.{part}")
        };
        self.row_list.store(props, &full);
    }
    pub fn load(&mut self, props: &BTreeMap<String, String>) {
        self.load_with_prepend(props, "", "");
    }
    pub fn load_with_prepend(
        &mut self,
        props: &BTreeMap<String, String>,
        prepend: &str,
        version: &str,
    ) {
        let part = self.hooks.get_load_prepend(version);
        let full = if prepend.is_empty() {
            part
        } else {
            format!("{prepend}.{part}")
        };
        self.row_list.load(props, &full);
    }
    pub fn is_scrolling(&self) -> bool {
        self.scrolling
    }
    pub fn set_tool_tip_text(&mut self) {}
    pub fn action_performed(&mut self) {}
    pub fn add_queue_table_listener(&mut self) {}
    pub fn send_queue_table_event(&mut self, event: QueueTableEvent) {
        self.queue_table_events.push(event);
    }
    pub fn remove_queue_table_listener(&mut self) {}
    pub fn queue_table_event_action(&mut self, event: QueueTableEvent) {
        self.row_list.queue_table_event_action(event);
    }
    pub fn secondary_queue_selected_action(&mut self) {
        self.row_list.secondary_queue_selected_action();
    }
    pub fn enable_queue_row(&mut self, name: &str, enable: bool) {
        self.row_list.enable_selection_field(name, enable);
    }
    pub fn enable_gpu_queue_rows(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Hooks;
    impl ProcessorTableHooks for Hooks {
        fn get_size(&self) -> usize {
            0
        }
        fn get_node(&self, _: usize) -> Option<String> {
            None
        }
        fn create_processor_table_row(&self, node: &str, _: usize) -> ProcessorTableRow {
            ProcessorTableRow::new(node)
        }
        fn get_header1_computer_text(&self) -> String {
            "Computer".into()
        }
        fn get_intermittent_command(&self, computer: &str) -> Option<String> {
            Some(computer.into())
        }
        fn is_exclude_node(&self, _: &str) -> bool {
            false
        }
        fn is_niceable(&self) -> bool {
            false
        }
        fn get_store_prepend(&self) -> String {
            "ProcessorTable".into()
        }
        fn get_load_prepend(&self, _: &str) -> String {
            "ProcessorTable".into()
        }
        fn init_row(&self, _: &mut ProcessorTableRow) {}
        fn is_queue_table(&self) -> bool {
            true
        }
        fn is_cpu_table(&self) -> bool {
            false
        }
        fn is_gpu_table(&self) -> bool {
            false
        }
        fn get_no_cpus_selected_error_message(&self) -> String {
            "No CPUs selected".into()
        }
    }
    #[test]
    fn row_list_contracts_to_selected_rows() {
        let mut table = ProcessorTable::new(Hooks, BTreeSet::new(), true, true, false);
        table.row_list.add(ProcessorTableRow::new("one"));
        table.row_list.add(ProcessorTableRow {
            selected: true,
            ..ProcessorTableRow::new("two")
        });
        table.row_list.add(ProcessorTableRow {
            secondary_queue_selected: true,
            ..ProcessorTableRow::new("three")
        });
        table.row_list.set_contracted_index(false);
        assert_eq!(table.size(), 2);
        table.build();
        assert!(!table.row_list.list[0].displayed);
        assert!(table.row_list.list[1].displayed && table.row_list.list[2].displayed);
    }
    #[test]
    fn map_load_and_persistence_follow_source_paths() {
        let mut table = ProcessorTable::new(Hooks, BTreeSet::new(), false, true, false);
        table.row_list.add(ProcessorTableRow::new("host"));
        table.set_computer_map(&BTreeMap::from([("host".into(), "4".into())]));
        table.set_load("host", 1., 5., 2, "alice");
        table.add_success("host");
        let mut props = BTreeMap::new();
        table.store(&mut props);
        assert_eq!(table.get_cpus_selected(), 4);
        assert_eq!(table.get_total_successes(), 1);
        assert_eq!(props["ProcessorTable.host.CPUsSelected"], "4");
    }
    #[test]
    fn state_changes_and_load_monitor_match_source_guards() {
        let mut table = ProcessorTable::new(Hooks, BTreeSet::new(), false, true, false);
        table.set_limited(true);
        assert_eq!(table.pack_count, 1);
        table.set_limited(true);
        assert_eq!(table.pack_count, 1);
        table.start_load();
        assert!(!table.is_stopped());
        table.set_secondary(true);
        table.end_load();
        assert!(table.is_stopped());
    }
}
