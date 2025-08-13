import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_model_download():
    """Тестирует загрузку модели DialoGPT-small"""
    
    print("🧪 Тестирую загрузку модели DialoGPT-small...")
    print("=" * 50)
    
    model_name = "microsoft/DialoGPT-small"
    
    try:
        print(f"📥 Загружаю токенизатор: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("✅ Токенизатор загружен успешно!")
        
        print(f"📥 Загружаю модель: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        print("✅ Модель загружена успешно!")
        
        # Проверяем размер модели
        model_size_mb = sum(p.numel() for p in model.parameters()) * 4 / (1024 * 1024)
        print(f"📊 Размер модели: {model_size_mb:.1f} МБ")
        
        # Проверяем, где сохранена модель
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        print(f"📁 Кэш Hugging Face: {cache_dir}")
        
        # Проверяем доступные модели в кэше
        if os.path.exists(cache_dir):
            print("📋 Найденные модели в кэше:")
            for item in os.listdir(cache_dir):
                if "microsoft" in item:
                    print(f"  - {item}")
        
        print("\n🎉 Тест загрузки модели прошел успешно!")
        print("Теперь можете запускать обучение!")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при загрузке модели: {e}")
        print("\n🔧 Возможные решения:")
        print("1. Проверьте интернет-соединение")
        print("2. Попробуйте запустить снова")
        print("3. Если проблема с памятью, закройте другие программы")
        return False

if __name__ == "__main__":
    test_model_download() 