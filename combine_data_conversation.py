import pandas as pd
import ast

def safe_eval(x):
    try:
        return ast.literal_eval(x)
    except:
        return []

def process_blended_skill_talk(url):
    df = pd.read_csv(url)

    data = []
    for _, row in df.iterrows():
        personas = safe_eval(row['personas'])
        persona = " ".join(personas) if personas else ""
        input_text = row['previous_utterance']
        output_text = safe_eval(row['free_messages'])
        output_text = output_text[0] if output_text else ""
        data.append({
            "persona": persona,
            "input": input_text,
            "output": output_text
        })
        blended_df = pd.DataFrame(data)
    return blended_df
def process_persional_chat(url):
    df = pd.read_csv(url)
    data =[]
    for _, row in df.iterrows():
        persona = row['Persona']
        output_text = row['chat']
        data.append({
            "persona": persona,
            "input": "",  # không có câu hỏi rõ ràng
            "output": output_text
        })

    persona_df = pd.DataFrame(data)
    return persona_df
def process_daily_dialog(url):
    df = pd.read_csv(url)

    def parse_dialog(x):
        x = x.strip()[1:-1]
        parts = [p.strip().strip("'").strip('"') for p in x.split('\n') if p.strip()]
        return parts

    data = []
    for _, row in df.iterrows():
        dialogs = parse_dialog(row['dialog'])
        if len(dialogs) < 2:
            continue
        for i in range(len(dialogs) - 1):
            input_text = dialogs[i]
            output_text = dialogs[i + 1]
            data.append({
                "persona": "",
                "input": input_text,
                "output": output_text
            })

    daily_df = pd.DataFrame(data)
    return daily_df

if __name__ == "__main__":
    blended_path = "data\\blended_skill_talk.csv"
    persona_path = "data\\archive\\personality.csv"
    daily_path = "D:\\AI_assistance\\data\\dailydialog.csv"

    blended_df = process_blended_skill_talk(blended_path)
    persona_df = process_persional_chat(persona_path)
    daily_df = process_daily_dialog(daily_path)
    # print(len(daily_df), len(blended_df), len(persona_df))

    combined_df = pd.concat([blended_df, persona_df, daily_df]).reset_index(drop=True)
    combined_df.to_csv("combined_conversation_dataset.csv", index=False)
    
