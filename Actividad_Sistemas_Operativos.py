import threading
import time

def tarea(nombre, delay):
    for i in range(3):
        print(f"[{nombre}] ejecución {i+1}")
        time.sleep(delay)

h1 = threading.Thread(target=tarea, args=("Hilo 1", 0.3))
h2 = threading.Thread(target=tarea, args=("Hilo 2", 0.5))
h3 = threading.Thread(target=tarea, args=("Hilo 3", 0.7))

h1.start()
h2.start()
h3.start()

h1.join()
h2.join()
h3.join()

print("✅ Todos los hilos terminaron.")