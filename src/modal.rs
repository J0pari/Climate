//! Canonical propositional modal logic over finite Kripke frames.
//!
//! This module gives scenario reasoning an actual modal-semantic foundation:
//! worlds, an accessibility relation, valuations of atomic propositions, and
//! compositional satisfaction for `not`, `and`, `or`, implication, necessity
//! (`box`), and possibility (`diamond`).
//!
//! Climate feasibility scores, transition probabilities, and physical models
//! are deliberately outside this kernel. A caller may use those to construct a
//! declared accessibility relation, but weighted plausibility is not itself the
//! definition of Kripke truth.

use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ModalError {
    #[error("a Kripke frame must contain at least one world")]
    EmptyFrame,
    #[error("world index {world} is outside frame with {world_count} worlds")]
    InvalidWorld { world: usize, world_count: usize },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KripkeFrame {
    successors: Vec<HashSet<usize>>,
}

impl KripkeFrame {
    pub fn new(
        world_count: usize,
        accessibility: impl IntoIterator<Item = (usize, usize)>,
    ) -> Result<Self, ModalError> {
        if world_count == 0 {
            return Err(ModalError::EmptyFrame);
        }
        let mut successors = vec![HashSet::new(); world_count];
        for (from, to) in accessibility {
            if from >= world_count {
                return Err(ModalError::InvalidWorld {
                    world: from,
                    world_count,
                });
            }
            if to >= world_count {
                return Err(ModalError::InvalidWorld {
                    world: to,
                    world_count,
                });
            }
            successors[from].insert(to);
        }
        Ok(Self { successors })
    }

    pub fn world_count(&self) -> usize {
        self.successors.len()
    }

    pub fn successors(&self, world: usize) -> Result<&HashSet<usize>, ModalError> {
        self.successors
            .get(world)
            .ok_or(ModalError::InvalidWorld {
                world,
                world_count: self.world_count(),
            })
    }

    pub fn is_reflexive(&self) -> bool {
        self.successors
            .iter()
            .enumerate()
            .all(|(world, next)| next.contains(&world))
    }

    pub fn is_symmetric(&self) -> bool {
        self.successors.iter().enumerate().all(|(from, next)| {
            next.iter()
                .all(|&to| self.successors[to].contains(&from))
        })
    }

    pub fn is_transitive(&self) -> bool {
        self.successors.iter().enumerate().all(|(from, next)| {
            next.iter().all(|&middle| {
                self.successors[middle]
                    .iter()
                    .all(|target| self.successors[from].contains(target))
            })
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Formula<A> {
    Atom(A),
    Falsum,
    Not(Box<Formula<A>>),
    And(Box<Formula<A>>, Box<Formula<A>>),
    Or(Box<Formula<A>>, Box<Formula<A>>),
    Implies(Box<Formula<A>>, Box<Formula<A>>),
    Necessarily(Box<Formula<A>>),
    Possibly(Box<Formula<A>>),
}

impl<A> Formula<A> {
    pub fn not(inner: Formula<A>) -> Self {
        Self::Not(Box::new(inner))
    }

    pub fn and(left: Formula<A>, right: Formula<A>) -> Self {
        Self::And(Box::new(left), Box::new(right))
    }

    pub fn or(left: Formula<A>, right: Formula<A>) -> Self {
        Self::Or(Box::new(left), Box::new(right))
    }

    pub fn implies(left: Formula<A>, right: Formula<A>) -> Self {
        Self::Implies(Box::new(left), Box::new(right))
    }

    pub fn necessarily(inner: Formula<A>) -> Self {
        Self::Necessarily(Box::new(inner))
    }

    pub fn possibly(inner: Formula<A>) -> Self {
        Self::Possibly(Box::new(inner))
    }
}

#[derive(Debug, Clone)]
pub struct KripkeModel<A>
where
    A: Eq + Hash + Clone,
{
    frame: KripkeFrame,
    valuation: HashMap<usize, HashSet<A>>,
}

impl<A> KripkeModel<A>
where
    A: Eq + Hash + Clone,
{
    pub fn new(
        frame: KripkeFrame,
        true_atoms: impl IntoIterator<Item = (usize, A)>,
    ) -> Result<Self, ModalError> {
        let mut valuation: HashMap<usize, HashSet<A>> = HashMap::new();
        for (world, atom) in true_atoms {
            if world >= frame.world_count() {
                return Err(ModalError::InvalidWorld {
                    world,
                    world_count: frame.world_count(),
                });
            }
            valuation.entry(world).or_default().insert(atom);
        }
        Ok(Self { frame, valuation })
    }

    pub fn frame(&self) -> &KripkeFrame {
        &self.frame
    }

    pub fn satisfies(&self, world: usize, formula: &Formula<A>) -> Result<bool, ModalError> {
        self.frame.successors(world)?;
        match formula {
            Formula::Atom(atom) => Ok(self
                .valuation
                .get(&world)
                .is_some_and(|atoms| atoms.contains(atom))),
            Formula::Falsum => Ok(false),
            Formula::Not(inner) => Ok(!self.satisfies(world, inner)?),
            Formula::And(left, right) => {
                Ok(self.satisfies(world, left)? && self.satisfies(world, right)?)
            }
            Formula::Or(left, right) => {
                Ok(self.satisfies(world, left)? || self.satisfies(world, right)?)
            }
            Formula::Implies(left, right) => {
                Ok(!self.satisfies(world, left)? || self.satisfies(world, right)?)
            }
            Formula::Necessarily(inner) => {
                for successor in self.frame.successors(world)? {
                    if !self.satisfies(*successor, inner)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            Formula::Possibly(inner) => {
                for successor in self.frame.successors(world)? {
                    if self.satisfies(*successor, inner)? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
        }
    }

    pub fn valid_everywhere(&self, formula: &Formula<A>) -> Result<bool, ModalError> {
        for world in 0..self.frame.world_count() {
            if !self.satisfies(world, formula)? {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn atom(name: &'static str) -> Formula<&'static str> {
        Formula::Atom(name)
    }

    fn valuations_for_two_atoms(world_count: usize) -> Vec<Vec<(usize, &'static str)>> {
        let slots = world_count * 2;
        (0usize..(1usize << slots))
            .map(|mask| {
                let mut valuation = Vec::new();
                for world in 0..world_count {
                    if mask & (1 << (2 * world)) != 0 {
                        valuation.push((world, "p"));
                    }
                    if mask & (1 << (2 * world + 1)) != 0 {
                        valuation.push((world, "q"));
                    }
                }
                valuation
            })
            .collect()
    }

    #[test]
    fn necessity_and_possibility_use_standard_accessibility_semantics() {
        let frame = KripkeFrame::new(3, [(0, 1), (0, 2), (1, 1), (2, 2)]).unwrap();
        let model = KripkeModel::new(frame, [(1, "safe"), (2, "safe"), (2, "warm")]).unwrap();

        assert!(model
            .satisfies(0, &Formula::necessarily(atom("safe")))
            .unwrap());
        assert!(model
            .satisfies(0, &Formula::possibly(atom("warm")))
            .unwrap());
        assert!(!model
            .satisfies(0, &Formula::necessarily(atom("warm")))
            .unwrap());
    }

    #[test]
    fn modal_k_axiom_holds_on_arbitrary_frames() {
        let frame = KripkeFrame::new(3, [(0, 1), (0, 2), (1, 2)]).unwrap();
        let k = Formula::implies(
            Formula::necessarily(Formula::implies(atom("p"), atom("q"))),
            Formula::implies(
                Formula::necessarily(atom("p")),
                Formula::necessarily(atom("q")),
            ),
        );

        for valuation in valuations_for_two_atoms(3) {
            let model = KripkeModel::new(frame.clone(), valuation).unwrap();
            assert!(model.valid_everywhere(&k).unwrap());
        }
    }

    #[test]
    fn reflexivity_is_exactly_what_supports_t_axiom_in_this_witness() {
        let reflexive = KripkeFrame::new(2, [(0, 0), (0, 1), (1, 1)]).unwrap();
        assert!(reflexive.is_reflexive());
        let t = Formula::implies(Formula::necessarily(atom("p")), atom("p"));

        for valuation in valuations_for_two_atoms(2) {
            let model = KripkeModel::new(reflexive.clone(), valuation).unwrap();
            assert!(model.valid_everywhere(&t).unwrap());
        }

        let non_reflexive = KripkeFrame::new(2, [(0, 1), (1, 1)]).unwrap();
        let countermodel = KripkeModel::new(non_reflexive, [(1, "p")]).unwrap();
        assert!(!countermodel.satisfies(0, &t).unwrap());
    }

    #[test]
    fn transitivity_supports_modal_4_and_missing_transitive_edge_has_countermodel() {
        let transitive = KripkeFrame::new(3, [(0, 1), (1, 2), (0, 2)]).unwrap();
        assert!(transitive.is_transitive());
        let four = Formula::implies(
            Formula::necessarily(atom("p")),
            Formula::necessarily(Formula::necessarily(atom("p"))),
        );

        for valuation in valuations_for_two_atoms(3) {
            let model = KripkeModel::new(transitive.clone(), valuation).unwrap();
            assert!(model.valid_everywhere(&four).unwrap());
        }

        let non_transitive = KripkeFrame::new(3, [(0, 1), (1, 2)]).unwrap();
        assert!(!non_transitive.is_transitive());
        let countermodel = KripkeModel::new(non_transitive, [(1, "p")]).unwrap();
        assert!(!countermodel.satisfies(0, &four).unwrap());
    }

    #[test]
    fn malformed_frames_and_valuations_fail_closed() {
        assert_eq!(KripkeFrame::new(0, []), Err(ModalError::EmptyFrame));
        assert_eq!(
            KripkeFrame::new(2, [(0, 2)]),
            Err(ModalError::InvalidWorld {
                world: 2,
                world_count: 2,
            })
        );

        let frame = KripkeFrame::new(2, [(0, 1)]).unwrap();
        let result = KripkeModel::new(frame, [(2, "p")]);
        assert!(matches!(
            result,
            Err(ModalError::InvalidWorld {
                world: 2,
                world_count: 2,
            })
        ));
    }
}
