import json
import os
import re
from pathlib import Path
from collections import defaultdict

def clean_text(text):
    """Очищает текст от эмодзи, ссылок и служебных сообщений"""
    if isinstance(text, list):
        text = ''.join([str(t['text']) if isinstance(t, dict) else str(t) for t in text])
    
    # Удаляем эмодзи и специальные символы
    text = re.sub(r'[^\w\s.,!?\-:;()]', ' ', str(text))
    # Удаляем множественные пробелы
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_chat_file(file_path, your_name="Andrey"):
    """Обрабатывает один JSON-чат и возвращает список диалогов"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        conversations = []
        current_context = []
        
        # Проверяем, что это личный чат
        if data.get('type') != 'personal_chat':
            return conversations
        
        for msg in data.get('messages', []):
            if msg.get('type') != 'message' or 'text' not in msg:
                continue
                
            sender = msg.get('from', '')
            text = clean_text(msg['text'])
            
            # Игнорируем короткие сообщения и служебные
            if not text or len(text) < 2 or text.startswith('http'):
                continue
                
            if sender == your_name:
                # Если это ваше сообщение - сохраняем как ответ
                if current_context:
                    conversations.append({
                        "context": current_context.copy(),
                        "response": text,
                        "source_file": os.path.basename(file_path),
                        "chat_name": data.get('name', 'Unknown')
                    })
                    current_context = []  # Сбрасываем контекст после вашего ответа
            else:
                # Сообщение собеседника добавляем в контекст
                current_context.append(text)
                # Ограничиваем длину контекста (последние 3 сообщения)
                if len(current_context) > 3:
                    current_context.pop(0)
        
        return conversations
        
    except Exception as e:
        print(f"Ошибка при обработке {file_path}: {e}")
        return []

def create_training_dataset():
    """Создает единый датасет из всех чатов"""
    
    # Путь к папке с чатами
    chats_folder = Path("2025_08_12")
    
    if not chats_folder.exists():
        print(f"Папка {chats_folder} не найдена!")
        return
    
    all_conversations = []
    total_files = 0
    processed_files = 0
    
    # Обрабатываем все JSON файлы
    for json_file in chats_folder.glob("*.json"):
        total_files += 1
        print(f"Обрабатываю: {json_file.name}")
        
        conversations = process_chat_file(json_file)
        if conversations:
            all_conversations.extend(conversations)
            processed_files += 1
            print(f"  -> Найдено {len(conversations)} диалогов")
        else:
            print(f"  -> Диалоги не найдены")
    
    print(f"\n=== РЕЗУЛЬТАТ ===")
    print(f"Обработано файлов: {processed_files}/{total_files}")
    print(f"Всего диалогов: {len(all_conversations)}")
    
    if all_conversations:
        # Сохраняем основной датасет
        output_file = "training_dataset.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_conversations, f, ensure_ascii=False, indent=2)
        print(f"Датасет сохранен в: {output_file}")
        
        # Создаем статистику
        stats = {
            "total_conversations": len(all_conversations),
            "processed_files": processed_files,
            "total_files": total_files,
            "chat_sources": {},
            "context_lengths": [],
            "response_lengths": []
        }
        
        for conv in all_conversations:
            # Статистика по источникам
            source = conv['source_file']
            stats['chat_sources'][source] = stats['chat_sources'].get(source, 0) + 1
            
            # Статистика по длине
            stats['context_lengths'].append(len(conv['context']))
            stats['response_lengths'].append(len(conv['response']))
        
        # Сохраняем статистику
        stats_file = "dataset_statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"Статистика сохранена в: {stats_file}")
        
        # Показываем примеры
        print(f"\n=== ПРИМЕРЫ ДИАЛОГОВ ===")
        for i, conv in enumerate(all_conversations[:3]):
            print(f"\nПример {i+1} (из {conv['source_file']}):")
            print(f"Контекст: {conv['context']}")
            print(f"Ваш ответ: {conv['response']}")
            print("-" * 50)
        
        # Рекомендации
        print(f"\n=== РЕКОМЕНДАЦИИ ===")
        if len(all_conversations) < 500:
            print("⚠️  Мало данных для обучения! Нужно минимум 500 диалогов.")
            print("   Рекомендуется собрать больше чатов или использовать data augmentation.")
        elif len(all_conversations) < 2000:
            print("✅ Достаточно данных для базового обучения (500-2000 диалогов)")
            print("   Модель будет работать, но качество может быть средним.")
        else:
            print("🎉 Отличное количество данных! (>2000 диалогов)")
            print("   Модель должна хорошо обучиться вашему стилю.")
        
        print(f"\nТеперь можете использовать файл '{output_file}' для обучения модели!")
        
    else:
        print("❌ Не удалось найти диалоги для обучения!")
        print("Проверьте:")
        print("1. Правильность структуры JSON файлов")
        print("2. Наличие сообщений от пользователя 'Andrey'")
        print("3. Формат экспорта из Telegram")

if __name__ == "__main__":
    print("🚀 Создание датасета для обучения модели...")
    print("=" * 50)
    create_training_dataset() 