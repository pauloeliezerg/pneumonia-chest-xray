# Demo de Inferência — Classificação de Pneumonia Pediátrica em Raios-X

Aplicação de inferência (Streamlit) do projeto acadêmico (Módulo 7 — PPI/SiDi).
Faça upload de uma radiografia de tórax pediátrica e receba a classificação
prevista (NORMAL / PNEUMONIA), a probabilidade associada e um mapa Grad-CAM.

⚠️ Projeto acadêmico. Não constitui dispositivo médico.

## Rodar localmente

Na raiz do repositório:

```bash
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py
```

Coloque o checkpoint `best_efficientnet_b3.pt` na raiz desta pasta antes de rodar.

## Demo pública

[https://\[TODO E2: usuario\]-pneumonia-chest-xray.streamlit.app](https://pneumonia-chest-xray.streamlit.app/)

Repositório completo: https://github.com/pauloeliezerg/pneumonia-chest-xray
