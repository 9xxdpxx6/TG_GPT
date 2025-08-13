#!/usr/bin/env python3
"""
Быстрый старт для TG_GPT проекта
Этот скрипт поможет быстро настроить и запустить проект
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def check_python_version():
    """Проверяет версию Python"""
    if sys.version_info < (3, 8):
        print("❌ Требуется Python 3.8 или выше!")
        print(f"Текущая версия: {sys.version}")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} - OK")
    return True

def check_dependencies():
    """Проверяет установленные зависимости"""
    required_packages = [
        'torch', 'transformers', 'datasets', 'numpy', 'pandas'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - установлен")
        except ImportError:
            print(f"❌ {package} - не установлен")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📦 Установите недостающие пакеты:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_data_files():
    """Проверяет наличие необходимых файлов"""
    data_dir = Path("dataset")
    chats_dir = data_dir / "2025_08_12"
    model_dir = data_dir / "my_dialogpt"
    
    print("\n📁 Проверка файлов проекта:")
    
    if not data_dir.exists():
        print("❌ Папка dataset/ не найдена")
        return False
    
    if not chats_dir.exists():
        print("❌ Папка dataset/2025_08_12/ не найдена")
        print("   Создайте папку и добавьте экспортированные JSON чаты")
        return False
    
    json_files = list(chats_dir.glob("*.json"))
    if not json_files:
        print("❌ В папке dataset/2025_08_12/ нет JSON файлов")
        print("   Экспортируйте чаты из Telegram в формате JSON")
        return False
    
    print(f"✅ Найдено {len(json_files)} JSON файлов чатов")
    
    if not model_dir.exists():
        print("⚠️  Папка с обученной моделью не найдена")
        print("   Запустите обучение: cd trainer && python train_model_locally_fixed.py")
        return False
    
    print("✅ Обученная модель найдена")
    return True

def prepare_dataset():
    """Подготавливает датасет для обучения"""
    print("\n🔄 Подготовка датасета...")
    
    script_path = Path("dataset/prepare_training_dataset.py")
    if not script_path.exists():
        print("❌ Скрипт prepare_training_dataset.py не найден")
        return False
    
    try:
        result = subprocess.run([
            sys.executable, str(script_path)
        ], capture_output=True, text=True, cwd="dataset")
        
        if result.returncode == 0:
            print("✅ Датасет успешно подготовлен")
            return True
        else:
            print(f"❌ Ошибка при подготовке датасета: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False

def test_model():
    """Тестирует обученную модель"""
    print("\n🧪 Тестирование модели...")
    
    script_path = Path("core/generator.py")
    if not script_path.exists():
        print("❌ Скрипт generator.py не найден")
        return False
    
    try:
        # Создаем простой тест
        test_script = """
import sys
sys.path.append('core')
from generator import MessageGenerator

try:
    generator = MessageGenerator()
    response = generator.generate_response(["Привет! Как дела?"])
    print(f"✅ Тест успешен! Ответ: {response}")
except Exception as e:
    print(f"❌ Ошибка теста: {e}")
"""
        
        with open("test_model.py", "w", encoding="utf-8") as f:
            f.write(test_script)
        
        result = subprocess.run([
            sys.executable, "test_model.py"
        ], capture_output=True, text=True)
        
        # Удаляем временный файл
        os.remove("test_model.py")
        
        if result.returncode == 0:
            print("✅ Модель работает корректно")
            return True
        else:
            print(f"❌ Ошибка теста: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False

def main():
    """Основная функция"""
    print("🚀 TG_GPT - Быстрый старт")
    print("=" * 40)
    
    # Проверяем Python
    if not check_python_version():
        return
    
    # Проверяем зависимости
    if not check_dependencies():
        print("\n💡 Установите зависимости:")
        print("pip install -r requirements.txt")
        return
    
    # Проверяем файлы
    if not check_data_files():
        return
    
    # Подготавливаем датасет
    if not prepare_dataset():
        return
    
    # Тестируем модель
    if not test_model():
        return
    
    print("\n🎉 Проект готов к использованию!")
    print("\n📋 Следующие шаги:")
    print("1. Настройте параметры в core/generator.py")
    print("2. Запустите генератор: python core/generator.py")
    print("3. Для Telegram бота создайте .env файл с токеном")
    print("4. Запустите бота: python bot.py")
    
    print("\n📚 Документация:")
    print("- README.md - основная документация")
    print("- trainer/README.md - обучение модели")
    print("- core/README.md - использование генератора")

if __name__ == "__main__":
    main()
