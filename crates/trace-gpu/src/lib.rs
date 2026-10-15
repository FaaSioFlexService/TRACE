//! Deteccion y planificacion del backend GPU.
//!
//! La deteccion no presupone NVIDIA/AMD/Intel: wgpu selecciona un adaptador
//! compatible con la plataforma y permite registrar fabricante, dispositivo,
//! backend y limites. Los kernels se mantendran separados del core CPU para
//! poder especializarlos por dense, CSR y ELLPACK.

pub mod shaders;

/// Informacion de un adaptador GPU detectado.
#[derive(Clone, Debug, serde::Serialize)]
pub struct GpuDeviceInfo {
    pub name: String,
    pub backend: String,
    pub device_type: String,
    pub driver: String,
    pub vendor: u32,
    pub device: u32,
}

/// Resultado de deteccion GPU.
#[derive(Clone, Debug, serde::Serialize)]
pub struct GpuDetection {
    pub available: bool,
    pub devices: Vec<GpuDeviceInfo>,
}

/// Detecta GPUs visibles mediante wgpu cuando la feature esta habilitada.
#[cfg(feature = "native-gpu")]
pub fn detect() -> GpuDetection {
    let instance = wgpu::Instance::new(
        &wgpu::InstanceDescriptor::new_without_display_handle(),
    );
    let adapters = instance.enumerate_adapters(wgpu::Backends::all());
    let devices = adapters
        .into_iter()
        .map(|adapter| {
            let info = adapter.get_info();
            GpuDeviceInfo {
                name: info.name,
                backend: format!("{:?}", info.backend),
                device_type: format!("{:?}", info.device_type),
                driver: info.driver,
                vendor: info.vendor,
                device: info.device,
            }
        })
        .collect::<Vec<_>>();
    GpuDetection { available: !devices.is_empty(), devices }
}

/// Devuelve una respuesta estable cuando el binario se compila sin soporte GPU.
#[cfg(not(feature = "native-gpu"))]
pub fn detect() -> GpuDetection {
    GpuDetection { available: false, devices: Vec::new() }
}
