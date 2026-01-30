import platform
import sys

import tensorflow as tf

print("=" * 70)
print("ПОЛНАЯ ПРОВЕРКА TENSORFLOW И GPU")
print("=" * 70)

# Системная информация
print(f"Система: {platform.system()} {platform.release()}")
print(f"Процессор: {platform.processor()}")
print(f"Python: {sys.version}")
print(f"TensorFlow: {tf.__version__}")

# Информация о сборке
print(f"\nИнформация о сборке TensorFlow:")
try:
    build_info = tf.sysconfig.get_build_info()
    print(f"  CUDA версия: {build_info.get('cuda_version', 'Неизвестно')}")
    print(f"  CUDNN версия: {build_info.get('cudnn_version', 'Неизвестно')}")
except:
    print("  Информация о сборке недоступна")

# Проверка GPU
print(f"\nПРОВЕРКА GPU:")
gpus = tf.config.list_physical_devices("GPU")

if gpus:
    print(f"✓ УСПЕХ: Найдено {len(gpus)} GPU устройств")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu.name}")
        print(f"    Тип: {gpu.device_type}")

    # Тест производительности
    print(f"\nТЕСТ ПРОИЗВОДИТЕЛЬНОСТИ:")
    try:
        import time

        import numpy as np

        # Тест на CPU для сравнения
        print("  Тест на CPU...")
        with tf.device("/CPU:0"):
            a_cpu = tf.random.normal([1000, 1000])
            b_cpu = tf.random.normal([1000, 1000])
            start = time.time()
            c_cpu = tf.matmul(a_cpu, b_cpu)
            cpu_time = time.time() - start

        # Тест на GPU
        print("  Тест на GPU...")
        with tf.device("/GPU:0"):
            a_gpu = tf.random.normal([1000, 1000])
            b_gpu = tf.random.normal([1000, 1000])
            start = time.time()
            c_gpu = tf.matmul(a_gpu, b_gpu)
            gpu_time = time.time() - start

        print(f"  Время CPU: {cpu_time:.3f} сек")
        print(f"  Время GPU: {gpu_time:.3f} сек")
        print(f"  Ускорение: {cpu_time/gpu_time:.1f}x")

    except Exception as e:
        print(f"  Ошибка при тесте: {e}")

else:
    print(f"✗ GPU не найдены")

    # Проверка всех устройств
    print(f"\nВсе доступные устройства:")
    for device in tf.config.list_physical_devices():
        print(f"  {device.device_type}: {device.name}")

print("\n" + "=" * 70)
