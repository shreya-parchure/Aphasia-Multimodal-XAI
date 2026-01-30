import pandas as pd
from sklearn.metrics import classification_report

def save_breakdowns_by_person_card(
    model,
    X,
    y,
    data_with_meta,
    person_col="SubjID",
    card_col="Agent_Noun",
    out_person_excel="performance_by_person.xlsx",
    out_card_excel="performance_by_card.xlsx"
):
    """
    Saves performance breakdowns by person and by card into two Excel files.
    """

    # --- Predict ---
    y_pred = model.predict(X)

    # --- Containers for results ---
    person_rows = []
    card_rows = []

    # ==========================
    # Breakdown by PERSON
    # ==========================
    persons = data_with_meta[person_col].unique()

    for person in persons:
        idx = data_with_meta[data_with_meta[person_col] == person].index

        y_true_p = y[idx]
        y_pred_p = y_pred[idx]

        # Get metrics as dictionary
        report_dict = classification_report(y_true_p, y_pred_p, output_dict=True)

        # Flatten into a row with prefix
        row = {"Person": person}
        for k, v in report_dict.items():
            if isinstance(v, dict):
                for metric, val in v.items():
                    row[f"{k}_{metric}"] = val
            else:
                row[k] = v

        person_rows.append(row)

    # Convert to DataFrame
    df_person = pd.DataFrame(person_rows)
    df_person.to_excel(out_person_excel, index=False)
    print(f"Saved: {out_person_excel}")


    # ==========================
    # Breakdown by CARD
    # ==========================
    cards = data_with_meta[card_col].unique()

    for card in cards:
        idx = data_with_meta[data_with_meta[card_col] == card].index

        y_true_c = y[idx]
        y_pred_c = y_pred[idx]

        report_dict = classification_report(y_true_c, y_pred_c, output_dict=True)

        row = {"Card": card}
        for k, v in report_dict.items():
            if isinstance(v, dict):
                for metric, val in v.items():
                    row[f"{k}_{metric}"] = val
            else:
                row[k] = v

        card_rows.append(row)

    df_card = pd.DataFrame(card_rows)
    df_card.to_excel(out_card_excel, index=False)
    print(f"Saved: {out_card_excel}")
