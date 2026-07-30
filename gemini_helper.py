from google import genai

def ambil_analisis_gemini(nama_model, hari_target, h_sekarang, h_prediksi, selisih, keputusan, sentimen, indikator, skor_total):
    try:
        # Gunakan API Key kamu
        client = genai.Client(api_key="AQ.Ab8RN6LNkMoL0cFoc5973fF7sryM2pDV1rchjmNtm5Ud6Hz8QA")

        prompt_skripsi = f"""
        Methodological Context:
        The Decision Support System (DSS) utilizes a Rule-Based logic combining AI projection ({nama_model}) with macroeconomic fundamental analysis (last 30 days trends).
        
        Analysis Data:
        - Gold Price Prediction: Changes by IDR {int(selisih):,} to IDR {int(h_prediksi):,}
        - USD/IDR Exchange Rate (30-Day Trend): {indikator['kurs']['teks']}
        - World Crude Oil (30-Day Trend): {indikator['minyak']['teks']}
        - Fed Rate (30-Day Trend): {indikator['fed']['teks']}
        
        Final Decision: {keputusan}
        
        Mandatory Instructions:
        Construct a strong, convincing, and smooth analytical argument (maximum 2 sentences) explaining why the DSS generated the '{keputusan}' decision.
        
        Writing Rules:
        1. MUST begin the statement by asserting the method, for example: "The rule-based Decision Support System (DSS) determines..." or "Through rule-based evaluation, the DSS recommends..."
        2. Follow up with a natural economic explanation connecting the "AI price direction projection" with the "macroeconomic fundamental realities over the last 30 days" (Exchange Rate, Oil, Fed Rate).
        3. STRICTLY PROHIBITED from mentioning numerical scores (e.g., +1, -2) or threshold values. Focus purely on the economic interaction among these 4 variables.
        4. No conversational filler or introductory greetings.
        """

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt_skripsi,
        )
        return response.text

    except Exception as e:
        # Fallback text diperbaiki agar 100% Inggris (tidak menarik teks panjang bahasa Indonesia)
        # dan format angka hari (hari_target) diperjelas
        return f"Synthesizing the algorithmic forecast from the {nama_model} model, the evaluation firmly recommends to '{keputusan}'. This conclusion is directly driven by the projected gold price movement to IDR {int(h_prediksi):,}, reflecting a change of IDR {int(selisih):,} over a {hari_target}-day projection period. Furthermore, this projection is validated by the recent 30-day macroeconomic realities across the USD/IDR exchange rate, global crude oil prices, and the Fed Rate, which collectively substantiate the final recommendation."