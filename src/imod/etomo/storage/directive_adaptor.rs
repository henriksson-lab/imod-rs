//! `IMOD/Etomo/src/etomo/storage/DirectiveAdaptor.java`.
//!
//! Adapts an autodoc statement to its directive.

use super::autodoc::statement::Statement;
use super::directive_def::DirectiveDef;

/// Java `public final class DirectiveAdaptor`.
#[derive(Default)]
pub struct DirectiveAdaptor {
    /// Java private `statement`, initially null.
    statement: Option<*mut dyn Statement>,
    /// Java private `directive`, initially null.
    directive: Option<String>,
    /// Java private `directiveDef`, initially null.
    directive_def: Option<DirectiveDef>,
}

impl DirectiveAdaptor {
    /// Java `DirectiveAdaptor()`.
    pub fn new() -> DirectiveAdaptor {
        DirectiveAdaptor::default()
    }

    /// Java public `set(ReadOnlyStatement)`.
    pub fn set(&mut self, statement: Option<*mut dyn Statement>) {
        self.statement = statement;
        self.reset();
    }

    /// Java private `reset()`.
    fn reset(&mut self) {
        self.directive = None;
        self.directive_def = None;
    }

    /// Java private `setup()`.
    fn setup(&mut self) {
        let Some(statement) = self.statement else {
            return;
        };
        if self.directive.is_some() && self.directive_def.is_some() {
            return;
        }
        // SAFETY: the statement belongs to an autodoc the registry keeps for the run.
        self.directive = unsafe { (*statement).get_left_side() };
        self.directive_def = DirectiveDef::get_instance(self.directive.as_deref());
    }

    /// Java public `getDirectiveDef()`.
    pub fn get_directive_def(&mut self) -> Option<DirectiveDef> {
        self.setup();
        self.directive_def
    }
}
