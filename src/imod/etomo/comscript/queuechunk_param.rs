//! `IMOD/Etomo/src/etomo/comscript/QueuechunkParam.java`.
//!
//! The intermittent command that asks a cluster queue for its load
//! (`bash <queuechunk> -a L`).

use super::intermittent_command::IntermittentCommand;
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::storage::network::Network;
use crate::imod::etomo::r#type::axis_id::AxisID;

/// Java public final `QueuechunkParam`.
pub struct QueuechunkParam {
    /// Java private final field `queue`.
    queue: Option<String>,
    /// Java private field `intermittentCommand`.
    intermittent_command: Option<String>,
}

impl QueuechunkParam {
    /// Java private `QueuechunkParam(String, AxisID, BaseManager)`.  The axis and
    /// manager are accepted and not kept, as in the source.
    fn new(
        queue: Option<&str>,
        _axis_id: AxisID,
        _manager: &'static dyn BaseManager,
    ) -> QueuechunkParam {
        QueuechunkParam {
            queue: queue.map(str::to_string),
            intermittent_command: None,
        }
    }

    /// Java public static `getLoadInstance(String, AxisID, BaseManager)`.
    pub fn get_load_instance(
        queue: Option<&str>,
        axis_id: AxisID,
        manager: &'static dyn BaseManager,
    ) -> QueuechunkParam {
        let mut instance = QueuechunkParam::new(queue, axis_id, manager);
        instance.set_intermittent_command(queue);
        instance
    }

    /// Java private `setIntermittentCommand(String)`.
    fn set_intermittent_command(&mut self, queue: Option<&str>) {
        let cluster = Network::get_queue(queue);
        if let Some(cluster) = cluster {
            self.intermittent_command = Some(format!(
                "bash {} -a L",
                cluster.get_command().unwrap_or("null".to_string())
            ));
        }
    }
}

impl IntermittentCommand for QueuechunkParam {
    /// Java `getInterval`.
    fn get_interval(&self) -> i32 {
        600000
    }

    /// Java `getComputer`.
    fn get_computer(&self) -> Option<String> {
        self.queue.clone()
    }

    /// Java `getEndCommand`.
    fn get_end_command(&self) -> Option<String> {
        None
    }

    /// Java `getIntermittentCommand`.  Returns intermittent command string or null
    /// if the queue name was not found by Network.
    fn get_intermittent_command(&self) -> Option<String> {
        self.intermittent_command.clone()
    }

    /// Java `getLocalStartCommand`.
    fn get_local_start_command(&self) -> Option<Vec<String>> {
        None
    }

    /// Java `getRemoteStartCommand`.
    fn get_remote_start_command(&self) -> Option<Vec<String>> {
        None
    }

    /// Java `notifySentIntermittentCommand`.
    fn notify_sent_intermittent_command(&self) -> bool {
        true
    }
}
