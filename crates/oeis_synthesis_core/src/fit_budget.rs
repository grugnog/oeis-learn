//! Cooperative slices within the caller's existing hard deadline.
use std::time::{Duration, Instant};

pub const POLICY: &str = "native-fit-deadlines/v1";

#[derive(Clone, Copy, Debug)]
pub struct FitBudget {
    pub search: Instant,
    pub production: Instant,
    pub response: Instant,
}

impl FitBudget {
    pub fn new(start: Instant, caller: Instant) -> Self {
        let response = caller.min(start + Duration::from_secs(2));
        let remaining = response.saturating_duration_since(start);
        let production = response - remaining / 20;
        let search = production - remaining / 4;
        Self { search, production, response }
    }
}
