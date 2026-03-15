#include <iostream>
#include "../include/obsia_api.h"

int main() {
    std::cout << "[TestRunner] Iniciando Obsia Mobile API..." << std::endl;

    ObsiaConfig config;
    config.model_path = "c:/Users/Luis Seoanes/Desktop/arrow/modeloFinal/models/qwen-medicina-q2k.gguf"; 
    config.rag_chunks_path = "c:/Users/Luis Seoanes/Desktop/arrow/modeloFinal/models/chunks.json"; 
    config.n_threads = 4; // Aprovechar mas hilos en PC para el test
    config.n_ctx = 1536;
    config.n_batch = 128;
    config.rag_k = 3;

    int init_res = obsia_init(&config);
    if (init_res != 0) {
        std::cerr << "[TestRunner] Error inicializando Obsia: " << init_res << std::endl;
        return 1;
    }

    std::cout << "[TestRunner] Obsia inicializado correctamente." << std::endl;
    std::cout << "[TestRunner] Probando consulta: 'tengo sangrado profuso despues del parto'" << std::endl;

    const char* response = obsia_chat("tengo sangrado profuso despues del parto");
    
    std::cout << "\n=== RESPUESTA ===\n" << response << "\n=================\n";

    obsia_free();
    std::cout << "[TestRunner] Recursos liberados. Test completado." << std::endl;
    return 0;
}
