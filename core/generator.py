import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MessageGenerator:
    """Генератор сообщений на основе обученной модели"""
    
    def __init__(self, model_path="../dataset/my_dialogpt"):
        """Инициализация генератора"""
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Используется устройство: {self.device}")
        self.load_model()
    
    def load_model(self):
        """Загружает обученную модель"""
        try:
            logger.info(f"Загружаю модель из {self.model_path}")
            
            if not torch.cuda.is_available():
                logger.warning("CUDA недоступна, используется CPU. Обучение будет медленнее.")
            
            # Загружаем токенизатор и модель
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32 if self.device == "cpu" else torch.float16
            )
            
            # Перемещаем модель на устройство
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("✅ Модель успешно загружена!")
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {e}")
            raise
    
    def generate_response(self, context_messages, max_length=100, temperature=0.7):
        """
        Генерирует ответ на основе контекста
        
        Аргументы:
        context_messages -- список последних сообщений собеседника
        max_length -- максимальная длина ответа
        temperature -- креативность (0.1-1.0, где 1.0 - самая креативная)
        
        Возвращает:
        Сгенерированный ответ
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Модель не загружена!")
        
        try:
            # Формируем входной текст
            dialog_text = ""
            for msg in context_messages:
                dialog_text += f"Собеседник: {msg}\n"
            dialog_text += "Вы: "
            
            logger.info(f"Генерирую ответ для контекста: {context_messages}")
            
            # Токенизируем
            inputs = self.tokenizer(
                dialog_text, 
                return_tensors="pt", 
                max_length=256, 
                truncation=True
            ).to(self.device)
            
            # Генерируем ответ
            with torch.no_grad():
                reply_ids = self.model.generate(
                    inputs["input_ids"],
                    max_length=inputs["input_ids"].shape[1] + max_length,
                    do_sample=True,
                    top_p=0.95,
                    top_k=50,
                    temperature=temperature,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,  # Избегаем повторений
                    no_repeat_ngram_size=3   # Избегаем повторения фраз
                )
            
            # Декодируем и очищаем ответ
            full_response = self.tokenizer.decode(reply_ids[0], skip_special_tokens=True)
            
            # Извлекаем только ваш ответ
            if "Вы: " in full_response:
                your_response = full_response.split("Вы: ")[-1].strip()
            else:
                your_response = full_response.strip()
            
            logger.info(f"Сгенерирован ответ: {your_response}")
            return your_response
            
        except Exception as e:
            logger.error(f"Ошибка при генерации: {e}")
            return "Извините, не удалось сгенерировать ответ."
    
    def generate_multiple_responses(self, context_messages, num_responses=3, temperature=0.7):
        """Генерирует несколько вариантов ответа"""
        responses = []
        
        for i in range(num_responses):
            # Немного варьируем температуру для разнообразия
            temp = temperature + (i * 0.1)
            response = self.generate_response(context_messages, temperature=temp)
            responses.append(response)
        
        return responses

def interactive_mode():
    """Интерактивный режим для тестирования генератора"""
    print("🤖 Генератор сообщений в вашем стиле")
    print("=" * 50)
    print("Введите сообщения собеседника (каждое с новой строки)")
    print("Когда закончите, оставьте строку пустой и нажмите Enter")
    print("Для выхода введите 'exit'")
    print("-" * 50)
    
    try:
        generator = MessageGenerator()
        
        while True:
            print("\n💬 Введите контекст:")
            context = []
            
            while True:
                msg = input("Сообщение собеседника: ").strip()
                if msg.lower() == 'exit':
                    print("👋 До свидания!")
                    return
                if not msg:
                    break
                context.append(msg)
            
            if not context:
                continue
            
            print(f"\n📝 Контекст: {context}")
            
            # Генерируем несколько вариантов
            print("\n🎯 Генерирую варианты ответов...")
            responses = generator.generate_multiple_responses(context, num_responses=3)
            
            print("\n✨ Варианты ответов:")
            for i, response in enumerate(responses, 1):
                print(f"{i}. {response}")
            
            # Позволяем выбрать лучший
            print("\n📋 Выберите лучший ответ (1-3) или нажмите Enter для пропуска:")
            choice = input("Ваш выбор: ").strip()
            
            if choice in ['1', '2', '3']:
                selected = responses[int(choice) - 1]
                print(f"\n✅ Выбран ответ: {selected}")
                print("📋 Скопируйте его и отправьте в Telegram!")
            else:
                print("⏭️  Ответ пропущен")
    
    except KeyboardInterrupt:
        print("\n\n👋 До свидания!")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        print("Убедитесь, что модель обучена и находится в папке '../dataset/my_dialogpt'")

def batch_mode(context_messages):
    """Пакетный режим для генерации одного ответа"""
    try:
        generator = MessageGenerator()
        response = generator.generate_response(context_messages)
        return response
    except Exception as e:
        logger.error(f"Ошибка в пакетном режиме: {e}")
        return f"Ошибка: {e}"

if __name__ == "__main__":
    print("🚀 Запуск генератора сообщений...")
    
    # Проверяем, есть ли обученная модель
    import os
    if not os.path.exists("../dataset/my_dialogpt"):
        print("❌ Обученная модель не найдена!")
        print("Сначала обучите модель, запустив:")
        print("python train_model_locally.py")
        print("\nИли запустите в интерактивном режиме для тестирования")
        choice = input("Запустить интерактивный режим? (y/n): ").lower()
        if choice == 'y':
            interactive_mode()
    else:
        interactive_mode() 