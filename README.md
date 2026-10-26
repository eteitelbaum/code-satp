# SATP Conflict Data Pipeline

Code and data for automatically coding conflict events from the [South Asia Terrorism Portal](https://www.satp.org) (SATP). We fine-tune language models on a hand-coded corpus of about 10,000 incident reports from India's Maoist insurgency (2005–2016) and compare what the models produce with what human coders produced.

The pipeline codes six aspects of each incident:

1. **Perpetrator:** who carried out the action (security forces, Maoists, unknown)
2. **Action type:** what was done (armed assault, bombing, infrastructure attack, abduction, arrest, surrender, seizure); an incident can have several
3. **Target type:** who or what was affected; an incident can have several
4. **Casualties:** deaths and injuries
5. **Other counts:** arrests, surrenders, abductions, property damage
6. **Location:** state, district, block, village, and other named places, geocoded to coordinates

## Repository layout

| Folder | Contents |
|---|---|
| `data/` | The hand-coded dataset, the R/Quarto scripts that clean it (`wrangle_satp.qmd`, `select_vars_for_models.qmd`), and the task-specific CSVs they produce |
| `documentation/` | The coding manual and manual coding procedure |
| `models/classification-models/` | Perpetrator (single-label) and action and target type (multi-label) classifiers: BERT, RoBERTa, DistilBERT, ConfliBERT, ELECTRA, XLNet, decoder-only LLMs, and class-imbalance strategies |
| `models/count-models/` | Casualty and count extraction with seq2seq models (T5 family) and LLMs |
| `models/location-models/` | Location extraction (seq2seq, span-based NER, LLMs) and geocoding experiments |
| `hugging-face-hosting-inference/` | Streamlit app that runs the production models on new incidents ([Hugging Face Space](https://huggingface.co/spaces/Mohiddin77/SATP-hosting-demo)) |
| `streamlit-app/` | Streamlit app for scraping new incidents from SATP |
| `village-matching/` | Matching coded village names to Census 2011 villages (SHRUG) |
| `papers/`, `presentations/` | Working papers and slides, with their results files |
| `legacy-notebooks/` | Earlier training and inference notebooks, kept for reference |

## Running the code

Model training runs in Google Colab on a GPU; each notebook clones the repository and loads its data from `data/`. To run the apps locally:

```bash
python -m venv venv
source venv/bin/activate
pip install -r hugging-face-hosting-inference/requirements.txt
cd hugging-face-hosting-inference && streamlit run app.py
```

The apps read from and write to Google Sheets and use the Google Maps API, so they need credentials in Streamlit secrets.

## Citation

The authors request that you cite the following paper in any published work that uses this code or data:

Teitelbaum, Emmanuel, Mohiddin Bacha Shaik, and Shiva Chakrabarty Sharma. 2026. "The Limits and Promise of Automated Event Coding: Evidence from the South Asia Terrorism Portal." SSRN Working Paper. <https://dx.doi.org/10.2139/ssrn.6163986>

```bibtex
@article{teitelbaum2026limits,
  title   = {The Limits and Promise of Automated Event Coding: Evidence from the South Asia Terrorism Portal},
  author  = {Teitelbaum, Emmanuel and Shaik, Mohiddin Bacha and Sharma, Shiva Chakrabarty},
  journal = {SSRN Working Paper},
  year    = {2026},
  doi     = {10.2139/ssrn.6163986}
}
```

GitHub's "Cite this repository" button gives the same reference.

## License

- **Code** is released under the [MIT License](LICENSE).
- **Our codings and model outputs** are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **Files derived from SHRUG** (e.g., `data/shrid_loc_names.csv`) carry SHRUG's license, [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Please also cite Asher, Lunt, Matsuura, and Novosad (2021), "Development Research at High Geographic Resolution: An Analysis of Night-Lights, Firms, and Poverty in India Using the SHRUG Open Data Platform," *World Bank Economic Review* 35(4).
- **Incident reports** come from the [South Asia Terrorism Portal](https://www.satp.org). See SATP's [terms of use](https://www.satp.org/disclaimer).
