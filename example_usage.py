#!/usr/bin/env python3
"""
Пример использования TG_GPT генератора сообщений
"""

import sys
from pathlib import Path

# Добавляем путь к модулю core
sys.path.append(str(Path(__file__).parent / "core"))

try:
    from generator import MessageGenerator
    
    def main():
        print("🤖 TG_GPT - Пример использования")
        print("=" * 40)
        
        # Инициализируем генератор
        print("📥 Загружаю модель...")
        generator = MessageGenerator()
        
        # Примеры диалогов
        examples = [
            ["Привет! Как дела?"],
            ["Что делаешь сегодня?"],
            ["Как прошел твой день?"],
            ["Расскажи что-нибудь интересное"],
            ["Какие у тебя планы на выходные?"]
        ]
        
        print("\n💬 Генерирую ответы...")
        print("-" * 40)
        
        for i, context in enumerate(examples, 1):
            print(f"\n{i}. Контекст: {' '.join(context)}")
            
            try:
                # Генерируем ответ
                response = generator.generate_response(
                    context_messages=context,
                    max_length=100,
                    temperature=0.7
                )
                
                print(f"   Ответ: {response}")
                
            except Exception as e:
                print(f"   ❌ Ошибка: {e}")
        
        print("\n" + "=" * 40)
        print("✅ Примеры завершены!")
        
        # Интерактивный режим
        print("\n🎯 Интерактивный режим (Ctrl+C для выхода)")
        print("Введите сообщение, на которое хотите получить ответ:")
        
        while True:
            try:
                user_input = input("\n👤 Вы: ").strip()
                if not user_input:
                    continue
                
                if user_input.lower() in ['выход', 'exit', 'quit']:
                    break
                
                # Генерируем ответ
                response = generator.generate_response(
                    context_messages=[user_input],
                    max_length=150,
                    temperature=0.8
                )
                
                print(f"🤖 Бот: {response}")
                
            except KeyboardInterrupt:
                print("\n\n👋 До свидания!")
                break
            except Exception as e:
                print(f"❌ Ошибка: {e}")
    
    if __name__ == "__main__":
        main()
        
except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")
    print("💡 Убедитесь, что:")
    print("1. Модель обучена и находится в dataset/my_dialogpt/")
    print("2. Установлены все зависимости: pip install -r requirements.txt")
    print("3. Скрипт запущен из корневой папки проекта")
except Exception as e:
    print(f"❌ Неожиданная ошибка: {e}")
