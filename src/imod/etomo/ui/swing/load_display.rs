//! `IMOD/Etomo/src/etomo/ui/swing/LoadDisplay.java`.
#![allow(dead_code)]

use super::processor_table::{ProcessorTable, ProcessorTableHooks};

/// Java `LoadDisplay`.
pub trait LoadDisplay {
    type ConstEtomoNumber: ToString;

    fn set_load(&mut self, computer: &str, load1: f64, load5: f64, users: i32, users_tooltip: &str);
    fn msg_load_failed(&mut self, computer: &str, reason: &str, tooltip: &str);
    fn msg_starting_process(
        &mut self,
        computer: &str,
        failure_reason1: &str,
        failure_reason2: &str,
    );
    fn set_cpu_usage(
        &mut self,
        computer: &str,
        cpu_usage: f64,
        number_of_processors: &Self::ConstEtomoNumber,
    );
    fn start_load(&mut self);
    fn stop_load(&mut self);
    fn end_load(&mut self);
    fn set_load_array(&mut self, computer: &str, load_array: &[String]);
}

impl<H: ProcessorTableHooks> LoadDisplay for ProcessorTable<H> {
    type ConstEtomoNumber = String;

    fn set_load(
        &mut self,
        computer: &str,
        load1: f64,
        load5: f64,
        users: i32,
        users_tooltip: &str,
    ) {
        ProcessorTable::set_load(self, computer, load1, load5, users, users_tooltip);
    }
    fn msg_load_failed(&mut self, computer: &str, reason: &str, tooltip: &str) {
        ProcessorTable::msg_load_failed(self, computer, reason, tooltip);
    }
    fn msg_starting_process(
        &mut self,
        computer: &str,
        failure_reason1: &str,
        failure_reason2: &str,
    ) {
        ProcessorTable::msg_starting_process(self, computer, failure_reason1, failure_reason2);
    }
    fn set_cpu_usage(&mut self, computer: &str, cpu_usage: f64, number_of_processors: &String) {
        ProcessorTable::set_cpu_usage(self, computer, cpu_usage, number_of_processors);
    }
    fn start_load(&mut self) {
        ProcessorTable::start_load(self);
    }
    fn stop_load(&mut self) {
        ProcessorTable::stop_load(self);
    }
    fn end_load(&mut self) {
        ProcessorTable::end_load(self);
    }
    fn set_load_array(&mut self, computer: &str, load_array: &[String]) {
        ProcessorTable::set_load_array(self, computer, load_array);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct Display {
        calls: Vec<&'static str>,
    }
    impl LoadDisplay for Display {
        type ConstEtomoNumber = i32;
        fn set_load(&mut self, _: &str, _: f64, _: f64, _: i32, _: &str) {
            self.calls.push("set_load");
        }
        fn msg_load_failed(&mut self, _: &str, _: &str, _: &str) {
            self.calls.push("failed");
        }
        fn msg_starting_process(&mut self, _: &str, _: &str, _: &str) {
            self.calls.push("starting");
        }
        fn set_cpu_usage(&mut self, _: &str, _: f64, _: &i32) {
            self.calls.push("cpu");
        }
        fn start_load(&mut self) {
            self.calls.push("start");
        }
        fn stop_load(&mut self) {
            self.calls.push("stop");
        }
        fn end_load(&mut self) {
            self.calls.push("end");
        }
        fn set_load_array(&mut self, _: &str, _: &[String]) {
            self.calls.push("array");
        }
    }
    #[test]
    fn load_contract_retains_all_eight_operations() {
        let mut display = Display::default();
        display.set_load("cpu", 1., 5., 2, "users");
        display.msg_load_failed("cpu", "why", "tip");
        display.msg_starting_process("cpu", "a", "b");
        display.set_cpu_usage("cpu", 50., &2);
        display.start_load();
        display.stop_load();
        display.end_load();
        display.set_load_array("cpu", &[]);
        assert_eq!(
            display.calls,
            [
                "set_load", "failed", "starting", "cpu", "start", "stop", "end", "array"
            ]
        );
    }
}
