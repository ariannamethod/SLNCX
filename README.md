# SLNCX (Wulf)

SLNCX stands for *Silencieux*. “Wulf1” is the call sign: wakes when summoned, works, исчезает.  
Think of the fixer from *Pulp Fiction* — nothing лишнего, только результат.  
Code lean. Intent precise.

**Arianna Method** underpins each exchange:  
Wulf слушает, отвечает сдержанно. Нет болтовни, нет суеты.  
Тишина — часть дизайна.

---

## Architecture

- **Grok1-inspired Mixture of Experts (MoE)**: 8 экспертов на слой, 2 активируются на каждый токен.
- **Контекстное окно** — до 8192 токенов.
- **Layer chaos**: 64 глубоких стека, разное маршрутизирование.
- **Rotary Position Embeddings (RoPE)** для длинного внимания.
- **2-bit Quantization** — модель помещается в память, работает на CPU.

NanoGPT-стиль inference, полностью без CUDA и облаков.

---

## Функциональность

- **CLI**:  
  Запусти:  
  ```bash
  python wulf_cli.py "prompt"

Быстро, без лишних слов.
	•	API:
Один endpoint /generate.
POST с JSON:

{ "user": "alice", "prompt": "Hello" }

Ответ — тоже JSON, просто и ясно.

	•	Логи:
Всё пишется в logs/wulf/ (JSONL).
Ошибки — в failures/.
	•	Checkpoints:
Файл весов указывай через CKPT_PATH или --ckpt. Гружу лениво, после первого запроса остаётся в памяти.
	•	Модулярность:
Всё разбито по слоям — attention, MoE, dense, всё можно менять и переписывать.

⸻

Running
	1.	Помести quantized checkpoint в out/ckpt.pt (или укажи путь через --ckpt).
	2.	Установи зависимости:
pip install -r requirements.txt
	3.	CLI-запрос:
python wulf_cli.py [--ckpt path/to/ckpt.pt] "your prompt"
	4.	Стартуй API:
uvicorn app:app --host 0.0.0.0 --port 8000

Без HuggingFace, без внешних сервисов.
Весит мало, работает быстро, молчит, пока не позовёшь.

⸻

Logging & Memory
	•	Логи диалогов — в logs/wulf/ (JSONL).
	•	Ошибки и трейсбеки — в failures/.
	•	В scripts/:
	•	session_logger.py — логирует пары prompt/response
	•	wulf_cli.py — CLI
	•	fail_log.py — сохраняет трейсбеки
	•	read_session_logs.py — читает и выводит логи

⸻

Model Components
	•	models/:
	•	layers/ — dense, decoder
	•	attention/ — multi-head с RoPE
	•	moe/ — эксперты и маршрутизация

⸻

Development
	•	Тесты:
pytest
	•	Линт:
ruff .

⸻

Deployment on Railway
	1.	Новый Railway-проект, укажи этот репозиторий.
	2.	Стартуй: python app.py
	3.	Загрузи свой out/ckpt.pt (asset/volume).
	4.	Дёргай /generate с JSON:

{
  "user": "alice",
  "prompt": "Hello"
}



⸻

Wulf всегда готов, но говорит только когда нужно.
Silence is the best answer.
