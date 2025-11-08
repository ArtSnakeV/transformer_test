from transformers import AutoTokenizer, AutoModelForCausalLM
import random as rnd

# Попередньо навчена модель
model_name = "gpt2"

# Завантаження токенізатора
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
# print(tokenizer)

# print(model)

#
# # Текст запиту (початок фрази)
# prompt = "Hello"
#
# # Перетворення тексту у вектор
# inputs = tokenizer(prompt, return_tensors="pt") # Перетворюємо prompt у вектор
#
# # Генерація продовження тексту
# outputs = model.generate(
#     **inputs,
#     max_new_tokens=50, # Кількість нових токенів, що ми можемо отримати від мережі
#     temperature=0.8, # Креативність
#     top_p=0.9, # Основна вибірка (контроль імовірності)
#     do_sample=True, # Дозволяє випадковість
#     repetition_penalty=1.2, # Уникнення повторів
# )
#
# # Декодування з токенів у текст
# generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True) # Відкидаємо спеціальні токени, лишаємо відповідь
#
# print(generated_text)


# Opening file
f = open("chatData.txt", "w") # Creating new file every time

while True:
    print("User:")
    prompt = input()
    if prompt == "exit" or prompt == "quit" or prompt == "q" or prompt == "e":
        break
    inputs = tokenizer(prompt, return_tensors="pt")  # Перетворюємо prompt у вектор
    # Генерація продовження тексту
    outputs = model.generate(
        **inputs,
        max_new_tokens=rnd.randint(30,70), # Кількість нових токенів, що ми можемо отримати від мережі
        temperature=0.8, # Креативність
        top_p=0.9, # Основна вибірка (контроль імовірності)
        do_sample=True, # Дозволяє випадковість
        repetition_penalty=1.2, # Уникнення повторів
    )
    # Декодування з токенів у текст
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True) # Відкидаємо спеціальні токени, лишаємо відповідь
    #
    print("Traveler: " + generated_text + "\n")
    f.write("User: " + prompt + "\n" + "AI: " + generated_text + "\n")
