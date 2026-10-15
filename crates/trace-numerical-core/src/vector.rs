//! Tipos vectoriales contiguos y operaciones auxiliares.

/// Vector numerico contiguo en precision f64.
#[derive(Clone, Debug, PartialEq)]
pub struct Vector {
    pub data: Vec<f64>,
}

impl Vector {
    /// Crea un vector a partir de almacenamiento existente.
    pub fn new(data: Vec<f64>) -> Self {
        Self { data }
    }

    /// Devuelve la longitud del vector.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Indica si el vector no contiene elementos.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}
