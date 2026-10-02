//! `IMOD/Etomo/src/etomo/storage/Network.java`.
//!
//! Description: Represents the queues and computers available in the local network.
//! Uses the cpu.adoc file, the IMOD_PROCESSORS environment variable, and the Settings
//! dialog (saved to the .etomo file).
//!
//! Copyright: Copyright 2009 - 2024 by the Regents of the University of Colorado
//!
//! Organization: Dept. of MCD Biology, University of Colorado
//!
//! Java's `Collection<Node>`/`Node` results are shared `Arc<Node>`s (see
//! `storage/node.rs`).

use super::cpu_adoc;
use super::node::{self, Node};
use crate::imod::etomo::base_manager::BaseManager;
use crate::imod::etomo::etomo_director;
use crate::imod::etomo::logic::processor_type::ProcessorType;
use crate::imod::etomo::process::system_program::SystemProgram;
use crate::imod::etomo::r#type::axis_id::AxisID;
use crate::imod::etomo::r#type::interface_type::InterfaceType;
use crate::imod::etomo::r#type::queue_mode::QueueMode;
use crate::imod::etomo::util::environment_variable;
use std::sync::{Arc, Mutex};

/// Java private static field `Hostname`, initialised to null.
static HOSTNAME: Mutex<Option<String>> = Mutex::new(None);

/// Java `Network`, a class of static methods (`private Network() {}`).
pub struct Network;

impl Network {
    /// Java static `getTotalCPUs`.  Returns the total available CPUs in the entire
    /// network.  Does not include queues.
    pub fn get_total_cpus(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        property_user_dir: Option<&str>,
    ) -> i32 {
        if cpu_adoc::INSTANCE.is_viable() {
            let nodes = cpu_adoc::INSTANCE.get_computers();
            if !nodes.is_empty() {
                let mut cores: i32 = 0;
                for node in nodes.iter() {
                    let cpus = node.get_cpus();
                    if !cpus.is_null() {
                        cores = cores.wrapping_add(cpus.get_int());
                    }
                }
                return cores;
            }
        }
        // When there isn't a cpu.adoc, use or generate a local host node.
        let local_host = Network::get_local_host(manager, axis_id, property_user_dir);
        if let Some(local_host) = local_host {
            let cpus = local_host.get_cpus();
            if !cpus.is_null() {
                return cpus.get_int();
            }
        }
        1
    }

    /// Java static `getTotalQueueCPUs`.  Returns the total available CPUs in all of the
    /// queues.  Does not include the Computer sections.
    pub fn get_total_queue_cpus(
        _manager: &'static dyn BaseManager,
        _axis_id: AxisID,
        _property_user_dir: Option<&str>,
    ) -> i32 {
        if cpu_adoc::INSTANCE.is_viable() {
            let nodes = cpu_adoc::INSTANCE.get_queues();
            let mut cores: i32 = 0;
            for node in nodes.iter() {
                let cpus = node.get_cpus();
                if !cpus.is_null() {
                    cores = cores.wrapping_add(cpus.get_int());
                }
            }
            return cores;
        }
        0
    }

    /// Java static `getQueueMaxCpus`.  Returns the queue with the largest number of
    /// available CPUs in the entire network.
    pub fn get_queue_max_cpus(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        property_user_dir: Option<&str>,
    ) -> i32 {
        if cpu_adoc::INSTANCE.is_viable() {
            let nodes = cpu_adoc::INSTANCE.get_queues();
            let mut cores: i32 = 0;
            let mut temp_cores: i32 = 0;
            for node in nodes.iter() {
                let cen_temp_cores = node.get_cpus();
                if !cen_temp_cores.is_null() {
                    temp_cores = cen_temp_cores.get_int();
                }
                if temp_cores > cores {
                    cores = temp_cores;
                }
            }
            return cores;
        }
        // When there isn't a cpu.adoc, use or generate a local host node.
        let local_host = Network::get_local_host(manager, axis_id, property_user_dir);
        if let Some(local_host) = local_host {
            let cpus = local_host.get_cpus();
            if !cpus.is_null() {
                return cpus.get_int();
            }
        }
        1
    }

    /// Java static `getLocalHost`.
    pub fn get_local_host(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        property_user_dir: Option<&str>,
    ) -> Option<Arc<Node>> {
        if cpu_adoc::INSTANCE.is_viable() {
            return cpu_adoc::INSTANCE.get_local_host_computer(manager, axis_id, property_user_dir);
        }
        if node::LOCAL_HOST_INSTANCE.lock().unwrap().is_none() {
            Node::create_local_instance(manager, axis_id, property_user_dir);
        }
        node::LOCAL_HOST_INSTANCE.lock().unwrap().clone()
    }

    /// Java static `getTotalGPUs`.  Returns the total available GPUs in the entire
    /// computer network.  Does not include queues.
    pub fn get_total_gpus(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        property_user_dir: Option<&str>,
    ) -> i32 {
        if cpu_adoc::INSTANCE.is_viable() {
            let nodes = cpu_adoc::INSTANCE.get_computers();
            let mut cores: i32 = 0;
            for node in nodes.iter() {
                cores =
                    cores.wrapping_add(node.get_total_gpus(manager, axis_id, property_user_dir));
            }
            return cores;
        }
        // When there isn't a cpu.adoc, use or generate a local host node.
        let local_host = Network::get_local_host(manager, axis_id, property_user_dir);
        if let Some(local_host) = local_host {
            return local_host.get_total_gpus(manager, axis_id, property_user_dir);
        }
        0
    }

    /// Java static `getLocalHostName`.  Gets the host name from the current host
    /// computer.  Calls hostname.
    pub fn get_local_host_name(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        property_user_dir: Option<&str>,
    ) -> Option<String> {
        if let Some(hostname) = HOSTNAME.lock().unwrap().as_ref() {
            return Some(hostname.clone());
        }
        let python_script_path = etomo_director::INSTANCE.get_python_script_path();
        let hostname = SystemProgram::new_array(
            Some(manager),
            property_user_dir.map(str::to_string),
            Some(vec![
                "python".to_string(),
                format!(
                    "{}b3dhostname",
                    python_script_path.as_deref().unwrap_or("null")
                ),
            ]),
            axis_id,
        );
        hostname.run();
        let stdout = hostname.get_std_output();
        let stdout = match stdout {
            Some(stdout) if !stdout.is_empty() => stdout,
            _ => return None,
        };
        *HOSTNAME.lock().unwrap() = Some(stdout[0].clone());
        Some(stdout[0].clone())
    }

    /// Java static `isParallelProcessingSetExternally`.  Parallel processing is set
    /// outside of Etomo if the cpu.adoc exists, or if the IMOD_PROCESSORS environment
    /// variable is set.
    pub fn is_parallel_processing_set_externally(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        property_user_dir: Option<&str>,
    ) -> bool {
        if cpu_adoc::INSTANCE.is_viable() {
            return true;
        }
        environment_variable::INSTANCE.exists(
            Some(manager),
            property_user_dir,
            "IMOD_PROCESSORS",
            Some(axis_id),
        )
    }

    /// Java static `isNonLocalOnlyGpuProcessingEnabled`.  Returns true if there are
    /// non-local-only GPU entries in the computer list.
    pub fn is_non_local_only_gpu_processing_enabled() -> bool {
        !cpu_adoc::INSTANCE.is_gpu_computer_list_empty(None)
    }

    /// Java static `isLocalHostGpuProcessingEnabled`.  Returns true if GPU processing is
    /// enabled in the cpu.adoc section for the current computer.  If cpu.adoc is not in
    /// use, returns true if GPU processing has been turned on in the Settings dialog.
    /// Otherwise returns false.  Side effect: May create Node.LOCAL_HOST_INSTANCE.
    pub fn is_local_host_gpu_processing_enabled(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        property_user_dir: Option<&str>,
    ) -> bool {
        let local_host = Network::get_local_host(manager, axis_id, property_user_dir);
        if let Some(local_host) = local_host {
            return local_host.is_gpu();
        }
        false
    }

    /// Java static `getLocalHostCpus`.
    ///
    /// Network.java:260 reads the CPUs from `Node.LOCAL_HOST_INSTANCE` rather than from
    /// the `localHost` it has just fetched; when cpu.adoc is viable `getLocalHost`
    /// returns the cpu.adoc section and `LOCAL_HOST_INSTANCE` is usually null, so this
    /// throws a NullPointerException (and otherwise reads the wrong node).  Fixed in
    /// translation: the CPUs are read from `localHost`.
    pub fn get_local_host_cpus(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        property_user_dir: Option<&str>,
    ) -> Option<i32> {
        let local_host = Network::get_local_host(manager, axis_id, property_user_dir);
        if let Some(local_host) = local_host {
            let cpus = local_host.get_cpus();
            if !cpus.is_null() {
                return Some(cpus.get_int());
            }
        }
        None
    }

    /// Java static `getLocalHostGPUs`.  Returns the number of GPUs from the local host
    /// section.  Returns null if there is no local host section.  If cpu.adoc is not
    /// viable, returns null because the only alternative is IMOD_PROCESSORS, which has
    /// no information about GPUs.
    pub fn get_local_host_gpus(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        property_user_dir: Option<&str>,
    ) -> Option<i32> {
        if cpu_adoc::INSTANCE.is_viable() {
            return cpu_adoc::INSTANCE.get_local_host_gpus(manager, axis_id, property_user_dir);
        }
        None
    }

    /// Java static `isNonLocalHostGpuProcessingEnabled`.  Returns false if cpu.adoc is
    /// missing, or if the only gpu enabled computer in cpu.adoc is the local host.
    pub fn is_non_local_host_gpu_processing_enabled(
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        property_user_dir: Option<&str>,
    ) -> bool {
        if !cpu_adoc::INSTANCE.is_viable() {
            return false;
        }
        let local_host =
            cpu_adoc::INSTANCE.get_local_host_computer(manager, axis_id, property_user_dir);
        !cpu_adoc::INSTANCE.is_gpu_computer_list_empty(local_host.as_ref())
            || !cpu_adoc::INSTANCE.is_gpu_queue_list_empty()
    }

    /// Java static `getNumComputers`.  Returns the number of Computer sections in
    /// cpu.adoc.  If cpu.adoc is not in use, returns 1.
    pub fn get_num_computers() -> i32 {
        if cpu_adoc::INSTANCE.is_viable() {
            let computer_list_size = cpu_adoc::INSTANCE.get_computer_list_size();
            if computer_list_size > 0 {
                return computer_list_size;
            }
        }
        // Count Node.LOCAL_INSTANCE if cpu.adoc is missing or has no computers.
        1
    }

    /// Java static `getComputer(BaseManager, int, AxisID, String)`.  Returns a Computer
    /// section from cpu.adoc by index, or returns Node.LOCAL_HOST_INSTANCE if cpu.adoc
    /// is not in use and the index is 0.  Otherwise returns null.  Side effect: may
    /// create Node.LOCAL_HOST_INSTANCE.
    pub fn get_computer(
        manager: &'static dyn BaseManager,
        index: i32,
        axis_id: AxisID,
        property_user_dir: Option<&str>,
    ) -> Option<Arc<Node>> {
        if cpu_adoc::INSTANCE.is_viable() {
            let node = cpu_adoc::INSTANCE.get_computer_by_index(index);
            if node.is_some() {
                return node;
            }
        }
        if index == 0 {
            if node::LOCAL_HOST_INSTANCE.lock().unwrap().is_none() {
                Node::create_local_instance(manager, axis_id, property_user_dir);
            }
            return node::LOCAL_HOST_INSTANCE.lock().unwrap().clone();
        }
        None
    }

    /// Java static `hasQueues()`.  Returns true if there are Queue sections in the
    /// cpu.adoc.
    pub fn has_queues() -> bool {
        !cpu_adoc::INSTANCE.is_queue_list_empty()
    }

    /// Java static `hasQueues(QueueMode)`.
    pub fn has_queues_mode(queue_mode: Option<QueueMode>) -> bool {
        cpu_adoc::INSTANCE.has_queues(queue_mode)
    }

    /// Java static `hasSecondaryQueues`.
    pub fn has_secondary_queues() -> bool {
        cpu_adoc::INSTANCE.has_secondary_queues()
    }

    /// Java static `getNumQueues`.  Gets the number of Queue sections in cpu.adoc.
    pub fn get_num_queues() -> i32 {
        cpu_adoc::INSTANCE.get_queue_list_size()
    }

    /// Java static `getQueue(String)`.  Gets a Queue section from cpu.adoc by name, or
    /// returns null.
    pub fn get_queue(name: Option<&str>) -> Option<Arc<Node>> {
        if let Some(name) = name {
            return cpu_adoc::INSTANCE.get_queue(name);
        }
        None
    }

    /// Java static `getQueue(int)`.  Gets a Queue section from cpu.adoc by index, or
    /// returns null.
    pub fn get_queue_by_index(index: i32) -> Option<Arc<Node>> {
        cpu_adoc::INSTANCE.get_queue_by_index(index)
    }

    /// Java static `isNumberGt1`.
    pub fn is_number_gt1(
        interface_type: Option<InterfaceType>,
        processor_type: ProcessorType,
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        property_user_dir: Option<&str>,
    ) -> bool {
        if !cpu_adoc::INSTANCE.is_viable() {
            let local_host = Network::get_local_host(manager, axis_id, property_user_dir);
            if let Some(local_host) = local_host {
                return local_host.is_number_gt1();
            }
            false
        } else {
            cpu_adoc::INSTANCE.is_number_gt1(interface_type, processor_type)
        }
    }

    /// Java static `isGpuGt1`.
    pub fn is_gpu_gt1(
        interface_type: Option<InterfaceType>,
        processor_type: ProcessorType,
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        property_user_dir: Option<&str>,
    ) -> bool {
        if !cpu_adoc::INSTANCE.is_viable() {
            let local_host = Network::get_local_host(manager, axis_id, property_user_dir);
            if let Some(local_host) = local_host {
                return local_host.is_gpu_gt1();
            }
            false
        } else {
            cpu_adoc::INSTANCE.is_gpu_gt1(interface_type, processor_type)
        }
    }

    /// Java static `isGpuAvailable`.
    pub fn is_gpu_available(
        computer_name: &str,
        manager: &'static dyn BaseManager,
        axis_id: AxisID,
        property_user_dir: Option<&str>,
    ) -> bool {
        if !cpu_adoc::INSTANCE.is_viable() {
            let local_host = Network::get_local_host(manager, axis_id, property_user_dir);
            if let Some(local_host) = local_host {
                return local_host.is_gpu();
            }
            return false;
        }
        let node = cpu_adoc::INSTANCE.get_computer(Some(computer_name));
        node.is_some_and(|node| {
            node.is_gpu()
                && (!node.is_gpu_local() || node.is_local_host(manager, axis_id, property_user_dir))
        })
    }

    /// Java static `isAnyQueueGpu`.
    pub fn is_any_queue_gpu() -> bool {
        let num_of_queues = Network::get_num_queues();
        for i in 0..num_of_queues {
            // Java dereferences `getQueue(i)` directly; every index below
            // `getNumQueues()` names a queue.
            if Network::get_queue_by_index(i).is_some_and(|queue| queue.is_gpu()) {
                return true;
            }
        }
        false
    }
}
