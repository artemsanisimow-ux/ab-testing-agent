"""
i18n.py — интернационализация для всех агентов
================================================
Использование:
    from i18n import t, get_language
    print(t("loading_tasks"))
    print(t("sprint_ready", count=5, sp=34))

Добавить в .env:
    LANGUAGE=en   # или ru
"""

import os
import argparse
from dotenv import load_dotenv

load_dotenv()

def _detect_language() -> str:
    """
    Определяет язык в порядке приоритета:
    1. Аргумент командной строки --lang
    2. Переменная окружения LANGUAGE в .env
    3. Дефолт: ru
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--lang", choices=["en", "ru"], default=None)
    args, _ = parser.parse_known_args()
    if args.lang:
        return args.lang
    env_lang = os.getenv("LANGUAGE", "ru").lower().strip()
    return env_lang if env_lang in ("en", "ru") else "ru"

LANGUAGE = _detect_language()

STRINGS = {

    # ─── Общие ───
    "yes": {"ru": "y", "en": "y"},
    "no": {"ru": "n", "en": "n"},
    "done": {"ru": "Готово", "en": "Done"},
    "error": {"ru": "Ошибка", "en": "Error"},
    "skip": {"ru": "пропустить", "en": "skip"},
    "warning": {"ru": "Предупреждение", "en": "Warning"},
    "session_id": {"ru": "Сессия", "en": "Session"},
    "continue_session": {"ru": "Продолжаю сессию", "en": "Resuming session"},
    "new_session": {"ru": "Новая сессия", "en": "New session"},

    # ─── Discovery агент ───
    "discovery_start": {"ru": "🚀 Discovery агент", "en": "🚀 Discovery agent"},
    "synthesizing": {"ru": "🔍 Синтезирую инсайты...", "en": "🔍 Synthesizing insights..."},
    "critiquing": {"ru": "🔎 Критикую черновик...", "en": "🔎 Critiquing draft..."},
    "refining": {"ru": "✏️  Улучшаю (итерация {n})...", "en": "✏️  Refining (iteration {n})..."},
    "finalizing": {"ru": "✅ Финализирую...", "en": "✅ Finalizing..."},
    "approve_prompt": {"ru": "✋ Утвердить? (y — да, n — доработать): ", "en": "✋ Approve? (y — yes, n — revise): "},
    "feedback_prompt": {"ru": "Что изменить? → ", "en": "What to change? → "},
    "approved": {"ru": "👍 Документ утверждён!", "en": "👍 Document approved!"},
    "report_saved": {"ru": "💾 Отчёт сохранён", "en": "💾 Report saved"},
    "audit_saved": {"ru": "📋 Audit сохранён", "en": "📋 Audit saved"},
    "missing_data": {"ru": "⚠️  НУЖНЫ ДОПОЛНИТЕЛЬНЫЕ ДАННЫЕ", "en": "⚠️  ADDITIONAL DATA NEEDED"},
    "missing_what": {"ru": "Чего не хватает:", "en": "What's missing:"},
    "add_data_prompt": {"ru": "Добавь данные и нажми Enter (или '{skip}' чтобы продолжить без них):", "en": "Add data and press Enter (or '{skip}' to continue without it):"},
    "user_skipped": {"ru": "Пользователь решил продолжить без дополнительных данных", "en": "User decided to continue without additional data"},
    "human_approved_note": {"ru": "Утверждено PM", "en": "Approved by PM"},
    "rejected_note": {"ru": "Отклонено PM", "en": "Rejected by PM"},

    # ─── Grooming агент ───
    "grooming_start": {"ru": "🚀 Grooming агент", "en": "🚀 Grooming agent"},
    "loading_tasks": {"ru": "📥 Загружаю задачи из Jira и Linear...", "en": "📥 Loading tasks from Jira and Linear..."},
    "jira_projects": {"ru": "Проекты Jira:", "en": "Jira projects:"},
    "linear_projects": {"ru": "Доступные проекты в Linear:", "en": "Available Linear projects:"},
    "select_project": {"ru": "Выбери номер проекта (или Enter чтобы пропустить): ", "en": "Select project number (or Enter to skip): "},
    "select_all": {"ru": "Выбери номер (или Enter чтобы загрузить все): ", "en": "Select number (or Enter to load all): "},
    "loaded_tasks": {"ru": "✅ Загружено: {jira} из Jira, {linear} из Linear", "en": "✅ Loaded: {jira} from Jira, {linear} from Linear"},
    "unique_tasks": {"ru": "   Уникальных задач: {n}", "en": "   Unique tasks: {n}"},
    "no_tasks_demo": {"ru": "⚠️  Задач не найдено. Используем демо-данные.", "en": "⚠️  No tasks found. Using demo data."},
    "tasks_for_grooming": {"ru": "Задачи для груминга:", "en": "Tasks for grooming:"},
    "task_header": {"ru": "📋 Задача {i}/{total}: {title}", "en": "📋 Task {i}/{total}: {title}"},
    "task_source": {"ru": "   Источник: {source} | Статус: {status}", "en": "   Source: {source} | Status: {status}"},
    "enriching": {"ru": "Уточняю описание...", "en": "Enriching description..."},
    "estimating": {"ru": "   📊 Оценка: {sp} SP | Уверенность: {confidence}", "en": "   📊 Estimate: {sp} SP | Confidence: {confidence}"},
    "splitting": {"ru": "   ✂️  Задача большая ({sp} SP) — разбиваю...", "en": "   ✂️  Task is large ({sp} SP) — splitting..."},
    "split_into": {"ru": "   Разбита на {n} подзадач:", "en": "   Split into {n} subtasks:"},
    "acceptance_done": {"ru": "   ✅ Критериев приёмки: {n}", "en": "   ✅ Acceptance criteria: {n}"},
    "priority_set": {"ru": "   🎯 Приоритет: {priority} | RICE: {score}", "en": "   🎯 Priority: {priority} | RICE: {score}"},
    "task_done": {"ru": "✅ Задача обработана: {title}", "en": "✅ Task processed: {title}"},
    "grooming_done": {"ru": "✅ Груминг завершён!", "en": "✅ Grooming complete!"},
    "groomed_count": {"ru": "   Обработано задач: {n}", "en": "   Tasks processed: {n}"},
    "clarification_needed": {"ru": "НУЖНО УТОЧНЕНИЕ", "en": "CLARIFICATION NEEDED"},
    "clarification_question": {"ru": "Вопрос: {q}", "en": "Question: {q}"},
    "current_estimate": {"ru": "Текущая оценка: {sp} SP | {confidence}", "en": "Current estimate: {sp} SP | {confidence}"},
    "accept_or_context": {"ru": "  y — принять как есть\n  [текст] — добавить контекст и пересчитать", "en": "  y — accept as is\n  [text] — add context and recalculate"},
    "linear_created": {"ru": "   ✨ Linear создано: {title}", "en": "   ✨ Linear created: {title}"},
    "linear_updated": {"ru": "   🔄 Linear обновлено: {title}", "en": "   🔄 Linear updated: {title}"},
    "jira_updated": {"ru": "   🔄 Jira обновлено: {key} — {title}", "en": "   🔄 Jira updated: {key} — {title}"},
    "jira_simplified": {"ru": "   ⚠️  Jira {key}: использован упрощённый payload (попытка {n})", "en": "   ⚠️  Jira {key}: used simplified payload (attempt {n})"},
    "jira_failed": {"ru": "   ⚠️  Jira update {key}: все попытки не удались ({code})", "en": "   ⚠️  Jira update {key}: all attempts failed ({code})"},
    "report_file": {"ru": "📄 Отчёт: {filename}", "en": "📄 Report: {filename}"},
    "audit_file": {"ru": "📋 Audit: {filename}", "en": "📋 Audit: {filename}"},

    # ─── Planning агент ───
    "planning_start": {"ru": "🚀 Planning Agent (Advanced)", "en": "🚀 Planning Agent (Advanced)"},
    "loading_data": {"ru": "📥 Загружаю данные из Jira и Linear...", "en": "📥 Loading data from Jira and Linear..."},
    "velocity_found": {"ru": "📊 Velocity samples: {samples}", "en": "📊 Velocity samples: {samples}"},
    "velocity_not_found": {"ru": "Не нашли velocity автоматически. Введи SP последних спринтов через запятую\n(или Enter для дефолта {default}): ", "en": "Velocity not found automatically. Enter SP of recent sprints separated by commas\n(or Enter for default {default}): "},
    "cycle_time": {"ru": "⏱️  Avg cycle time: {ct} дней | Throughput: {tp} задач/спринт", "en": "⏱️  Avg cycle time: {ct} days | Throughput: {tp} tasks/sprint"},
    "backlog_loaded": {"ru": "✅ Бэклог: {n} задач | Tech debt: {td} ({ratio}%)", "en": "✅ Backlog: {n} tasks | Tech debt: {td} ({ratio}%)"},
    "team_capacity_prompt": {"ru": "👥 Введи доступность команды на спринт ({days} дней):\n   Формат: Имя:дней (или просто Enter для дефолта)", "en": "👥 Enter team availability for sprint ({days} days):\n   Format: Name:days (or just Enter for default)"},
    "wip_limit": {"ru": "🔄 WIP limit (Little's Law): {n} задач", "en": "🔄 WIP limit (Little's Law): {n} tasks"},
    "monte_carlo": {"ru": "🎲 Monte Carlo симуляция ({n:,} итераций)...", "en": "🎲 Monte Carlo simulation ({n:,} iterations)..."},
    "mc_p50": {"ru": "   P50 velocity: {v} SP", "en": "   P50 velocity: {v} SP"},
    "mc_p85": {"ru": "   P85 velocity: {v} SP", "en": "   P85 velocity: {v} SP"},
    "mc_prob1": {"ru": "   Вероятность завершить за 1 спринт: {p}%", "en": "   Probability to complete in 1 sprint: {p}%"},
    "mc_prob2": {"ru": "   Вероятность завершить за 2 спринта: {p}%", "en": "   Probability to complete in 2 sprints: {p}%"},
    "mc_recommended": {"ru": "   Рекомендуемый commitment (P85 × 0.9): {sp} SP", "en": "   Recommended commitment (P85 × 0.9): {sp} SP"},
    "debt_health": {"ru": "   🔧 Tech debt health score: {score}/100", "en": "   🔧 Tech debt health score: {score}/100"},
    "prioritized": {"ru": "   📊 Приоритизировано: {n} задач", "en": "   📊 Prioritized: {n} tasks"},
    "selected_tasks": {"ru": "   📋 Отобрано: {n} задач ({sp} SP)", "en": "   📋 Selected: {n} tasks ({sp} SP)"},
    "tech_debt_sp": {"ru": "   🔧 Tech debt: {sp} SP ({pct}%)", "en": "   🔧 Tech debt: {sp} SP ({pct}%)"},
    "wip_batches": {"ru": "   🔄 WIP batches: {n}", "en": "   🔄 WIP batches: {n}"},
    "excluded": {"ru": "   ❌ Исключено: {n} задач", "en": "   ❌ Excluded: {n} tasks"},
    "go_nogo": {"ru": "   🎯 Go/NoGo: {verdict} | Уверенность: {conf}%", "en": "   🎯 Go/NoGo: {verdict} | Confidence: {conf}%"},
    "sprint_goal_label": {"ru": "   🎯 {goal}", "en": "   🎯 {goal}"},
    "stretch_goal_label": {"ru": "   ⭐ Stretch: {goal}", "en": "   ⭐ Stretch: {goal}"},
    "sprint_plan_header": {"ru": "✋ ПЛАН СПРИНТА — ТРЕБУЕТ ПОДТВЕРЖДЕНИЯ", "en": "✋ SPRINT PLAN — REQUIRES APPROVAL"},
    "goal_label": {"ru": "🎯 ЦЕЛЬ: {goal}", "en": "🎯 GOAL: {goal}"},
    "metrics_label": {"ru": "📊 МЕТРИКИ:", "en": "📊 METRICS:"},
    "sp_metrics": {"ru": "   SP: {total} из {available} доступных ({pct}% загрузка)", "en": "   SP: {total} of {available} available ({pct}% load)"},
    "prob_metrics": {"ru": "   Вероятность завершить: {p1}% за 1 спринт | {p2}% за 2", "en": "   Probability to complete: {p1}% in 1 sprint | {p2}% in 2"},
    "tasks_label": {"ru": "📋 ЗАДАЧИ ({n}):", "en": "📋 TASKS ({n}):"},
    "wip_batches_label": {"ru": "🔄 WIP БАТЧИ (параллельная работа):", "en": "🔄 WIP BATCHES (parallel work):"},
    "risks_label": {"ru": "⚠️  ТОП РИСКИ (pre-mortem):", "en": "⚠️  TOP RISKS (pre-mortem):"},
    "early_warning": {"ru": "   Ранний сигнал: {w}", "en": "   Early warning: {w}"},
    "mitigation": {"ru": "   Митигация: {m}", "en": "   Mitigation: {m}"},
    "success_metrics_label": {"ru": "✅ МЕТРИКИ УСПЕХА:", "en": "✅ SUCCESS METRICS:"},
    "approve_options": {"ru": "y — утвердить | n — изменить состав | r — изменить SP", "en": "y — approve | n — change tasks | r — change SP"},
    "your_choice": {"ru": "Твой выбор: ", "en": "Your choice: "},
    "new_sp_prompt": {"ru": "Новое количество SP: ", "en": "New SP count: "},
    "change_prompt": {"ru": "Что изменить? → ", "en": "What to change? → "},
    "sprint_ready": {"ru": "✅ Спринт готов!\n   {n} задач | {sp} SP | {verdict}", "en": "✅ Sprint ready!\n   {n} tasks | {sp} SP | {verdict}"},
    "publish_prompt": {"ru": "📤 Опубликовать спринт в Jira и Linear? (y/n): ", "en": "📤 Publish sprint to Jira and Linear? (y/n): "},
    "sprint_name_prompt": {"ru": "Название спринта (Enter для авто): ", "en": "Sprint name (Enter for auto): "},
    "activate_prompt": {"ru": "Активировать спринт в Jira сразу? (y/n): ", "en": "Activate sprint in Jira immediately? (y/n): "},
    "session_saved": {"ru": "💾 Session ID: {sid}", "en": "💾 Session ID: {sid}"},

    # ─── Sprint Publisher ───
    "publishing_header": {"ru": "📤 ПУБЛИКУЮ ПЛАН СПРИНТА", "en": "📤 PUBLISHING SPRINT PLAN"},
    "publishing_jira": {"ru": "🚀 Публикую спринт в Jira...", "en": "🚀 Publishing sprint to Jira..."},
    "publishing_linear": {"ru": "🚀 Публикую Cycle в Linear...", "en": "🚀 Publishing Cycle to Linear..."},
    "jira_not_configured": {"ru": "Jira не настроена — проверь .env", "en": "Jira not configured — check .env"},
    "linear_not_configured": {"ru": "Linear не настроен — проверь .env", "en": "Linear not configured — check .env"},
    "board_found": {"ru": "   📋 Доска: {name} (ID: {id})", "en": "   📋 Board: {name} (ID: {id})"},
    "sprint_created": {"ru": "   ✅ Спринт создан: {name} (ID: {id})", "en": "   ✅ Sprint created: {name} (ID: {id})"},
    "sprint_create_error": {"ru": "   ❌ Ошибка создания спринта: {e}", "en": "   ❌ Sprint creation error: {e}"},
    "issues_added": {"ru": "   ✅ Добавлено задач в спринт: {n}", "en": "   ✅ Tasks added to sprint: {n}"},
    "sp_updated": {"ru": "   ✅ SP обновлено: {n}/{total} задач", "en": "   ✅ SP updated: {n}/{total} tasks"},
    "sprint_activated": {"ru": "   ✅ Спринт активирован", "en": "   ✅ Sprint activated"},
    "jira_sprint_done": {"ru": "   🎉 Jira: спринт '{name}' создан (ID: {id})", "en": "   🎉 Jira: sprint '{name}' created (ID: {id})"},
    "cycle_created": {"ru": "   ✅ Cycle создан: {name} (ID: {id})", "en": "   ✅ Cycle created: {name} (ID: {id})"},
    "cycle_create_error": {"ru": "   ❌ Ошибка создания Cycle: {e}", "en": "   ❌ Cycle creation error: {e}"},
    "issues_in_cycle": {"ru": "   ✅ Задач добавлено в Cycle: {n}/{total}", "en": "   ✅ Tasks added to Cycle: {n}/{total}"},
    "estimates_updated": {"ru": "   ✅ Estimates обновлено: {n}/{total}", "en": "   ✅ Estimates updated: {n}/{total}"},
    "linear_cycle_done": {"ru": "   🎉 Linear: Cycle '{name}' создан (ID: {id})", "en": "   🎉 Linear: Cycle '{name}' created (ID: {id})"},
    "publish_result": {"ru": "📊 РЕЗУЛЬТАТ:", "en": "📊 RESULT:"},
    "jira_ok": {"ru": "   Jira: ✅ OK", "en": "   Jira: ✅ OK"},
    "jira_error": {"ru": "   Jira: ❌ Ошибка", "en": "   Jira: ❌ Error"},
    "linear_ok": {"ru": "   Linear: ✅ OK", "en": "   Linear: ✅ OK"},
    "linear_error": {"ru": "   Linear: ❌ Ошибка", "en": "   Linear: ❌ Error"},

    # ─── Linear Sync ───
    "linear_team": {"ru": "   👥 Linear team: {name} ({id})", "en": "   👥 Linear team: {name} ({id})"},
    "syncing_linear": {"ru": "🔄 Синхронизирую {n} задач в Linear...", "en": "🔄 Syncing {n} tasks to Linear..."},
    "sync_stats": {"ru": "   ✅ Создано: {created} | Обновлено: {updated} | Пропущено: {skipped} | Ошибок: {errors}", "en": "   ✅ Created: {created} | Updated: {updated} | Skipped: {skipped} | Errors: {errors}"},

    # ─── Jira Sync ───
    "syncing_jira": {"ru": "🔄 Обновляю {n} задач в Jira...", "en": "🔄 Updating {n} tasks in Jira..."},
    "jira_sync_stats": {"ru": "   ✅ Обновлено: {updated} | Пропущено: {skipped} | Ошибок: {errors}", "en": "   ✅ Updated: {updated} | Skipped: {skipped} | Errors: {errors}"},

    # ─── Промпты (для модели) ───
    "prompt_synthesize": {"ru": "Синтезируй инсайты.", "en": "Synthesize insights."},
    "prompt_critique": {"ru": "Проверь.", "en": "Check it."},
    "prompt_refine": {"ru": "Улучши.", "en": "Improve it."},
    "prompt_finalize": {"ru": "Выполни задачу и верни JSON.", "en": "Execute the task and return JSON."},
    "prompt_enrich": {"ru": "Выполни задачу и верни JSON.", "en": "Execute the task and return JSON."},
    "prompt_estimate": {"ru": "Выполни задачу и верни JSON.", "en": "Execute the task and return JSON."},
    "prompt_accept": {"ru": "Выполни задачу и верни JSON.", "en": "Execute the task and return JSON."},
    "prompt_prioritize": {"ru": "Выполни задачу и верни JSON.", "en": "Execute the task and return JSON."},
}


def t(msg_key: str, **kwargs) -> str:
    """
    Возвращает строку на текущем языке.

    Пример:
        t("task_header", i=1, total=9, title="Fix bug")
        t("loaded_tasks", jira=5, linear=3)
    """
    entry = STRINGS.get(msg_key)
    if not entry:
        return f"[{msg_key}]"  # Fallback — видно что строка не найдена

    text = entry.get(LANGUAGE, entry.get("en", f"[{msg_key}]"))

    if kwargs:
        try:
            text = text.format(**kwargs)
        except KeyError:
            pass  # Если не все аргументы переданы — возвращаем как есть

    return text


def get_language() -> str:
    return LANGUAGE


def set_language(lang: str):
    """Программно сменить язык (для тестов)."""
    global LANGUAGE
    if lang in ("en", "ru"):
        LANGUAGE = lang


def get_language_instruction() -> str:
    """
    Возвращает инструкцию для модели отвечать на нужном языке.
    Вставляется в начало каждого системного промпта.
    """
    if LANGUAGE == "en":
        return "IMPORTANT: You must respond entirely in English. All text, headings, labels, and content must be in English.\n\n"
    else:
        return "ВАЖНО: Отвечай полностью на русском языке. Весь текст, заголовки, метки и содержимое должны быть на русском.\n\n"
