import json
import os
import torch
from pathlib import Path
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_training_data(dataset_path="training_dataset.json"):
    """Загружает подготовленный датасет"""
    logger.info(f"Загружаю датасет из {dataset_path}")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Файл {dataset_path} не найден! Сначала запустите prepare_training_dataset.py")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Загружено {len(data)} диалогов")
    return data

def prepare_dataset_for_training(data):
    """Подготавливает датасет для обучения"""
    logger.info("Подготавливаю датасет для обучения...")
    
    # Преобразуем в формат Dataset
    dataset = Dataset.from_list(data)
    
    # Загружаем токенизатор
    model_name = "microsoft/DialoGPT-small"
    logger.info(f"Загружаю токенизатор: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    def format_dataset(examples):
        """Форматирует данные для обучения"""
        inputs = []
        for i in range(len(examples['context'])):
            context = examples['context'][i]
            response = examples['response'][i]
            
            # Формируем текст в формате диалога
            dialog_text = ""
            for msg in context:
                dialog_text += f"Собеседник: {msg}\n"
            dialog_text += f"Вы: {response}{tokenizer.eos_token}"
            
            inputs.append(dialog_text)
        
        # Токенизируем с ограничением длины
        model_inputs = tokenizer(
            inputs, 
            max_length=256,  # Уменьшаем для экономии памяти
            truncation=True, 
            padding="max_length"
        )
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        return model_inputs
    
    # Применяем форматирование
    logger.info("Токенизирую датасет...")
    tokenized_dataset = dataset.map(
        format_dataset, 
        batched=True, 
        remove_columns=dataset.column_names
    )
    
    # Разделяем на обучение и валидацию
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1)
    logger.info(f"Разделено на {len(split_dataset['train'])} обучающих и {len(split_dataset['test'])} тестовых примеров")
    
    return tokenized_dataset, split_dataset, tokenizer

def load_model_and_tokenizer(model_name="microsoft/DialoGPT-small"):
    """Загружает модель и токенизатор"""
    logger.info(f"Загружаю модель: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Добавляем специальный токен
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def setup_training_arguments(output_dir="./results"):
    """Настраивает параметры обучения для локального компьютера"""
    logger.info("Настраиваю параметры обучения для локального компьютера...")
    
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,                    # Начинаем с 2 эпох
        per_device_train_batch_size=2,         # Маленький batch size для экономии памяти
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,         # Экономия памяти
        warmup_steps=50,                       # Быстрый старт
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=25,                      # Чаще логи для контроля
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",                      # Отключаем внешние сервисы
        save_total_limit=2,                    # Ограничиваем количество сохранений
        dataloader_pin_memory=False,           # Экономия памяти
        dataloader_num_workers=0,              # Для Windows
    )

def train_model(model, tokenizer, train_dataset, eval_dataset, training_args):
    """Обучает модель"""
    logger.info("Начинаю обучение модели...")
    
    # Создаем data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Создаем Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Запуск обучения
    logger.info("🚀 Запускаю обучение...")
    logger.info("Это может занять 1-3 часа на локальном компьютере")
    logger.info("Можете оставить компьютер работать и вернуться позже")
    
    trainer.train()
    
    return trainer

def save_model(trainer, tokenizer, output_dir="./my_dialogpt"):
    """Сохраняет обученную модель"""
    logger.info(f"Сохраняю модель в {output_dir}")
    
    # Создаем директорию
    os.makedirs(output_dir, exist_ok=True)
    
    # Сохраняем модель
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logger.info(f"✅ Модель сохранена в {output_dir}")
    logger.info("Теперь можете использовать её для генерации сообщений!")

def main():
    """Основная функция"""
    print("🤖 Локальное обучение модели DialoGPT на ваших данных")
    print("=" * 60)
    
    try:
        # 1. Загружаем данные
        data = load_training_data()
        
        # 2. Подготавливаем датасет
        tokenized_dataset, split_dataset, tokenizer = prepare_dataset_for_training(data)
        
        # 3. Загружаем модель
        model, tokenizer = load_model_and_tokenizer()
        
        # 4. Настраиваем параметры обучения
        training_args = setup_training_arguments()
        
        # 5. Обучаем модель
        trainer = train_model(
            model, tokenizer, 
            split_dataset["train"], 
            split_dataset["test"], 
            training_args
        )
        
        # 6. Сохраняем результат
        save_model(trainer, tokenizer)
        
        print("\n🎉 Обучение завершено успешно!")
        print("📁 Модель сохранена в папке './my_dialogpt'")
        print("🔧 Теперь можете запустить generator.py для тестирования!")
        
    except Exception as e:
        logger.error(f"Ошибка при обучении: {e}")
        print(f"\n❌ Произошла ошибка: {e}")
        print("Проверьте:")
        print("1. Установлены ли все библиотеки (pip install transformers torch datasets)")
        print("2. Есть ли файл training_dataset.json")
        print("3. Достаточно ли места на диске")

if __name__ == "__main__":
    main() 