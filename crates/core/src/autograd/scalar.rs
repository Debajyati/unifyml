use crate::autograd::AutogradError;

pub type NodeId = usize;

#[derive(Debug, Clone, Copy)]
pub enum Op {
    Leaf,
    Add(NodeId, NodeId),
    Sub(NodeId, NodeId),
    Mul(NodeId, NodeId),
    Div(NodeId, NodeId),
    Neg(NodeId),
    Exp(NodeId),
    Ln(NodeId),
    Powf(NodeId, f32),
}

#[derive(Debug, Clone)]
pub struct Node {
    value: f32,
    grad: f32,
    op: Op,
}

#[derive(Debug, Default)]
pub struct Tape {
    nodes: Vec<Node>,
}

impl Tape {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    pub fn var(&mut self, value: f32) -> NodeId {
        let id = self.nodes.len();
        self.nodes.push(Node {
            value,
            grad: 0.0,
            op: Op::Leaf,
        });
        id
    }

    /// Safely retrieves the forward-pass value for a specific node
    pub fn value(&self, id: NodeId) -> Result<f32, AutogradError> {
        self.nodes
            .get(id)
            .map(|n| n.value)
            .ok_or(AutogradError::InvalidNodeId(id))
    }

    pub fn add(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId, AutogradError> {
        let val_lhs = self.value(lhs)?;
        let val_rhs = self.value(rhs)?;
        let id = self.nodes.len();
        self.nodes.push(Node {
            value: val_lhs + val_rhs,
            grad: 0.0,
            op: Op::Add(lhs, rhs),
        });
        Ok(id)
    }

    pub fn sub(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId, AutogradError> {
        let val_lhs = self.value(lhs)?;
        let val_rhs = self.value(rhs)?;
        let id = self.nodes.len();
        self.nodes.push(Node {
            value: val_lhs - val_rhs,
            grad: 0.0,
            op: Op::Sub(lhs, rhs),
        });
        Ok(id)
    }

    pub fn mul(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId, AutogradError> {
        let val_lhs = self.value(lhs)?;
        let val_rhs = self.value(rhs)?;
        let id = self.nodes.len();
        self.nodes.push(Node {
            value: val_lhs * val_rhs,
            grad: 0.0,
            op: Op::Mul(lhs, rhs),
        });
        Ok(id)
    }

    pub fn div(&mut self, lhs: NodeId, rhs: NodeId) -> Result<NodeId, AutogradError> {
        let val_lhs = self.value(lhs)?;
        let val_rhs = self.value(rhs)?;
        let id = self.nodes.len();
        self.nodes.push(Node {
            value: val_lhs / val_rhs,
            grad: 0.0,
            op: Op::Div(lhs, rhs),
        });
        Ok(id)
    }

    pub fn neg(&mut self, x: NodeId) -> Result<NodeId, AutogradError> {
        let val = self.value(x)?;
        let id = self.nodes.len();
        self.nodes.push(Node {
            value: -val,
            grad: 0.0,
            op: Op::Neg(x),
        });
        Ok(id)
    }

    pub fn exp(&mut self, x: NodeId) -> Result<NodeId, AutogradError> {
        let val = self.value(x)?;
        let id = self.nodes.len();
        self.nodes.push(Node {
            value: val.exp(),
            grad: 0.0,
            op: Op::Exp(x),
        });
        Ok(id)
    }

    pub fn ln(&mut self, x: NodeId) -> Result<NodeId, AutogradError> {
        let val = self.value(x)?;
        let id = self.nodes.len();
        self.nodes.push(Node {
            value: val.ln(),
            grad: 0.0,
            op: Op::Ln(x),
        });
        Ok(id)
    }

    pub fn powf(&mut self, x: NodeId, p: f32) -> Result<NodeId, AutogradError> {
        let val = self.value(x)?;
        let id = self.nodes.len();
        self.nodes.push(Node {
            value: val.powf(p),
            grad: 0.0,
            op: Op::Powf(x, p),
        });
        Ok(id)
    }

    pub fn grad(&self, id: NodeId) -> Result<f32, AutogradError> {
        self.nodes
            .get(id)
            .map(|n| n.grad)
            .ok_or(AutogradError::InvalidNodeId(id))
    }

    pub fn backward(&mut self, root: NodeId) -> Result<(), AutogradError> {
        if root >= self.nodes.len() {
            return Err(AutogradError::InvalidNodeId(root));
        }

        for node in &mut self.nodes {
            node.grad = 0.0;
        }

        self.nodes[root].grad = 1.0;

        for i in (0..=root).rev() {
            let upstream_grad = self.nodes[i].grad;
            let op = self.nodes[i].op;
            let current_val = self.nodes[i].value;

            match op {
                Op::Leaf => {}
                Op::Add(lhs, rhs) => {
                    self.nodes[lhs].grad += upstream_grad;
                    self.nodes[rhs].grad += upstream_grad;
                }
                Op::Sub(lhs, rhs) => {
                    self.nodes[lhs].grad += upstream_grad;
                    self.nodes[rhs].grad -= upstream_grad;
                }
                Op::Mul(lhs, rhs) => {
                    let val_lhs = self.nodes[lhs].value;
                    let val_rhs = self.nodes[rhs].value;
                    self.nodes[lhs].grad += upstream_grad * val_rhs;
                    self.nodes[rhs].grad += upstream_grad * val_lhs;
                }
                Op::Div(lhs, rhs) => {
                    let val_lhs = self.nodes[lhs].value;
                    let val_rhs = self.nodes[rhs].value;
                    self.nodes[lhs].grad += upstream_grad * (1.0 / val_rhs);
                    self.nodes[rhs].grad += upstream_grad * (-val_lhs / (val_rhs * val_rhs));
                }
                Op::Neg(x) => {
                    self.nodes[x].grad -= upstream_grad;
                }
                Op::Exp(x) => {
                    // Optimized: The derivative of exp(x) is exp(x), which we already
                    // calculated and stored in current_val during the forward pass!
                    self.nodes[x].grad += upstream_grad * current_val;
                }
                Op::Ln(x) => {
                    let val_x = self.nodes[x].value;
                    self.nodes[x].grad += upstream_grad * (1.0 / val_x);
                }
                Op::Powf(x, p) => {
                    let val_x = self.nodes[x].value;
                    self.nodes[x].grad += upstream_grad * (p * val_x.powf(p - 1.0));
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod scalar_autograd_tests {
    use super::*;

    #[test]
    fn test_shared_dependency_accumulation() {
        let mut tape = Tape::new();
        // z = x * x, where x = 3
        let x = tape.var(3.0);
        let z = tape.mul(x, x).unwrap();

        assert_eq!(tape.nodes[z].value, 9.0);

        tape.backward(z).unwrap();

        // dz/dx = 2x = 6.0
        // Because x was used twice in the mul operation, its gradient
        // must accumulate the contributions from both dependency edges.
        assert_eq!(tape.grad(x).unwrap(), 6.0);
    }

    #[test]
    fn test_repeated_backward_calls() {
        let mut tape = Tape::new();
        let x = tape.var(2.0);
        let y = tape.var(3.0);
        let z = tape.add(x, y).unwrap();

        // First backward pass
        tape.backward(z).unwrap();
        assert_eq!(tape.grad(x).unwrap(), 1.0);

        // Second backward pass should cleanly reset and recompute
        tape.backward(z).unwrap();
        assert_eq!(tape.grad(x).unwrap(), 1.0); // Would be 2.0 if not cleared!
    }

    #[test]
    fn test_multi_operation_chain() {
        let mut tape = Tape::new();
        // Compute z = (x - y) / (x + y)
        // With x = 2.0, y = 1.0 -> z = 1/3
        let x = tape.var(2.0);
        let y = tape.var(1.0);

        let num = tape.sub(x, y).unwrap(); // 2 - 1 = 1
        let den = tape.add(x, y).unwrap(); // 2 + 1 = 3
        let z = tape.div(num, den).unwrap();

        assert_eq!(tape.nodes[z].value, 1.0 / 3.0);

        tape.backward(z).unwrap();

        // Analytical derivatives:
        // dz/dx = 2y / (x + y)^2 = 2(1) / (3)^2 = 2/9
        // dz/dy = -2x / (x + y)^2 = -2(2) / (3)^2 = -4/9

        let expected_dx = 2.0 / 9.0;
        let expected_dy = -4.0 / 9.0;

        assert!((tape.grad(x).unwrap() - expected_dx).abs() < 1e-6);
        assert!((tape.grad(y).unwrap() - expected_dy).abs() < 1e-6);
    }

    #[test]
    fn test_negation_and_invalid_ids() {
        let mut tape = Tape::new();
        let x = tape.var(5.0);
        let neg_x = tape.neg(x).unwrap();

        tape.backward(neg_x).unwrap();
        assert_eq!(tape.grad(x).unwrap(), -1.0);

        // Verify invalid IDs return cleanly rather than panicking
        assert_eq!(tape.add(x, 999), Err(AutogradError::InvalidNodeId(999)));
        assert_eq!(tape.backward(999), Err(AutogradError::InvalidNodeId(999)));
    }

    #[test]
    fn test_gradient_accumulation_across_paths() {
        let mut tape = Tape::new();

        let x = tape.var(2.0);
        let y = tape.mul(x, x).unwrap(); // x²
        let z = tape.add(y, x).unwrap(); // x² + x

        tape.backward(z).unwrap();

        // dz/dx = 2x + 1 = 5.0
        assert_eq!(tape.grad(x).unwrap(), 5.0);
    }

    #[test]
    fn test_finite_difference_check() {
        // We will evaluate f(x) = ln(x^3) / exp(x) at x = 2.0
        let x_val = 2.0;

        // 1. Analytical/Autograd computation
        let mut tape = Tape::new();
        let x = tape.var(x_val);

        let x_cubed = tape.powf(x, 3.0).unwrap();
        let num = tape.ln(x_cubed).unwrap();
        let den = tape.exp(x).unwrap();
        let z = tape.div(num, den).unwrap();

        tape.backward(z).unwrap();
        let autograd_dx = tape.grad(x).unwrap();

        // 2. Finite difference approximation
        let epsilon = 1e-4;

        let f = |val: f32| -> f32 { val.powf(3.0).ln() / val.exp() };

        let f_plus = f(x_val + epsilon);
        let f_minus = f(x_val - epsilon);
        let numerical_dx = (f_plus - f_minus) / (2.0 * epsilon);

        // 3. Compare with a small tolerance
        assert!(
            (autograd_dx - numerical_dx).abs() < 1e-4,
            "Autograd: {}, Numerical: {}",
            autograd_dx,
            numerical_dx
        );
    }

    /// Reusable test utility to verify autograd against numerical finite differences
    fn check_gradient_1d<F, G>(x_val: f32, f_analytical: F, build_graph: G)
    where
        F: Fn(f32) -> f32,
        G: Fn(&mut Tape, NodeId) -> NodeId,
    {
        let mut tape = Tape::new();
        let x = tape.var(x_val);
        let z = build_graph(&mut tape, x);

        tape.backward(z).unwrap();
        let autograd_dx = tape.grad(x).unwrap();

        // 1. Change epsilon to 1e-3 for better f32 stability
        let epsilon = 1e-3;
        let num_dx =
            (f_analytical(x_val + epsilon) - f_analytical(x_val - epsilon)) / (2.0 * epsilon);

        // 2. Relax tolerance to 1e-2 for f32 comparisons
        assert!(
            (autograd_dx - num_dx).abs() < 1e-2,
            "Mismatch at x={}: autograd={}, numeric={}",
            x_val,
            autograd_dx,
            num_dx
        );
    }

    #[test]
    fn test_exp_gradient() {
        let mut tape = Tape::new();
        let x = tape.var(0.0);
        let y = tape.exp(x).unwrap();

        tape.backward(y).unwrap();

        // d/dx [exp(0)] = exp(0) = 1.0
        assert_eq!(tape.grad(x).unwrap(), 1.0);
    }

    #[test]
    fn test_ln_gradient() {
        let mut tape = Tape::new();
        let x = tape.var(2.0);
        let y = tape.ln(x).unwrap();

        tape.backward(y).unwrap();

        // d/dx [ln(2)] = 1/2 = 0.5
        assert!((tape.grad(x).unwrap() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_powf_gradients() {
        let mut tape = Tape::new();

        // powf(x, 3) at x=2 -> dx = 3*(2^2) = 12
        let x1 = tape.var(2.0);
        let y1 = tape.powf(x1, 3.0).unwrap();
        tape.backward(y1).unwrap();
        assert_eq!(tape.grad(x1).unwrap(), 12.0);

        // powf(x, 0) at x=2 -> dx = 0*(2^-1) = 0
        let x2 = tape.var(2.0);
        let y2 = tape.powf(x2, 0.0).unwrap();
        tape.backward(y2).unwrap();
        assert_eq!(tape.grad(x2).unwrap(), 0.0);

        // Edge case note: Evaluating tape.powf(0.0, p) will currently result in NaN or Inf
        // gradients if p < 1.0, because the local derivative computes 0.0.powf(p - 1.0).
        // This maps cleanly to standard IEEE-754 floating point behavior.
    }

    #[test]
    fn test_finite_difference_utility() {
        // Test a compound function: f(x) = sin(x) conceptually, but using our primitives
        // Let's do f(x) = exp(x) * ln(x) at x = 2.0
        check_gradient_1d(
            2.0,
            |x| x.exp() * x.ln(),
            |tape, x| {
                let exp_x = tape.exp(x).unwrap();
                let ln_x = tape.ln(x).unwrap();
                tape.mul(exp_x, ln_x).unwrap()
            },
        );
    }
}
