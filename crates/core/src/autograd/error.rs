use crate::autograd::scalar::NodeId;

#[derive(Debug, PartialEq, Eq)]
pub enum AutogradError {
    InvalidNodeId(NodeId),
}
