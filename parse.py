import os
import spacy

version = '1.0'

text = "COVID-19 is a coronavirus outbreak that initially appeared in Wuhan, Hubei Province, China, in December 2019, but it has already evolved into a pandemic spreading rapidly worldwide(1,2). As of 18 March 2020, a total number of 194909 cases of COVID-19 have been reported, including 7876 deaths, the majority of which have been reported in China (3242) and Italy (2505)(3). However, as the pandemic is still unfortunately under progression, there are limited data with regard to the clinical characteristics of the patients as well as to their prognostic factors(4). Smoking, to date, has been assumed to be possibly associated with adverse disease prognosis, as extensive evidence has highlighted the negative impact of tobacco use on lung health and its causal association with a plethora of respiratory diseases(5). Smoking is also detrimental to the immune system and its responsiveness to infections, making smokers more vulnerable to infectious diseases(6). Previous studies have shown that smokers are twice more likely than non-smokers to contract influenza and have more severe symptoms, while smokers were also noted to have higher mortality in the previous MERS-CoV outbreak(7,8). Given the gap in the evidence, we conducted a systematic review of studies on COVID-19 that included information on patients’ smoking status to evaluate the association between smoking and COVID-19 outcomes including the severity of the disease, the need for mechanical ventilation, the need for intensive care unit (ICU) hospitalization and death. The literature search was conducted on 17 March 2020, using two databases (PubMed, ScienceDirect), with the search terms: [‘smoking’ OR ‘tobacco’ OR ‘risk factors’ OR ‘smoker*’] AND [‘COVID-19’ OR ‘COVID 19’ OR ‘novel coronavirus’ OR ‘sars cov-2’ OR ‘sars cov 2’] and included studies published in 2019 and 2020. Further inclusion criteria were that the studies were in English and referred to humans. We also searched the reference lists of the studies included. A total of 71 studies were retrieved through the search, of which 66 were excluded after full-text screening, leaving five studies that were included. All of the studies were conducted in China, four in Wuhan and one across provinces in mainland China. The populations in all studies were patients with COVID-19, and the sample size ranged from 41 to 1099 patients. With regard to the study design, retrospective and prospective methods were used, and the timeframe of all five studies covered the first two months of the COVID-19 pandemic (December 2019, January 2020). Specifically, Zhou et al.(9) studied the epidemiological characteristics of 191 individuals infected with COVID-19, without, however, reporting in more detail the mortality risk factors and the clinical outcomes of the disease. Among the 191 patients, there were 54 deaths, while 137 survived. Among those that died, 9% were current smokers compared to 4% among those that survived, with no statistically significant difference between the smoking rates of survivors and non-survivors (p=0.21) with regard to mortality from COVID-19. Similarly, Zhang et al.(10) presented clinical characteristics of 140 patients with COVID-19. The results showed that among severe patients (n=58), 3.4% were current smokers and 6.9% were former smokers, in contrast to non-severe patients (n=82) among which 0% were current smokers and 3.7% were former smokers , leading to an OR of 2.23; (95% CI: 0.65–7.63; p=0.2). Huang et al.(11) studied the epidemiological characteristics of COVID-19 among 41 patients. In this study, none of those who needed to be admitted to an ICU (n=13) was a current smoker. In contrast, three patients from the non-ICU group were current smokers, with no statistically significant difference between the two groups of patients (p=0.31), albeit the small sample size of the study. The largest study population of 1099 patients with COVID-19 was provided by Guan et al.(12) from multiple regions of mainland China. Descriptive results on the smoking status of patients were provided for the 1099 patients, of which 173 had severe symptoms, and 926 had non-severe symptoms. Among the patients with severe symptoms, 16.9% were current smokers and 5.2% were former smokers, in contrast to patients with non-severe symptoms where 11.8% were current smokers and 1.3% were former smokers. Additionally, in the group of patients that either needed mechanical ventilation, admission to an ICU or died, 25.5% were current smokers and 7.6% were former smokers. In contrast, in the group of patients that did not have these adverse outcomes, only 11.8% were current smokers and 1.6% were former smokers. No statistical analysis for evaluating the association between the severity of the disease outcome and smoking status was conducted in that study. Finally, Liu et al.(13) found among their population of 78 patients with COVID-19 that the adverse outcome group had a significantly higher proportion of patients with a history of smoking (27.3%) than the group that showed improvement or stabilization (3.0%), with this difference statistically significant at the p=0.018 level. In their multivariate logistic regression analysis, the history of smoking was a risk factor of disease progression (OR=14.28; 95% CI: 1.58–25.00; p= 0.018). We identified five studies that reported data on the smoking status of patients infected with COVID-19. Notably, in the largest study that assessed severity, there were higher percentages of current and former smokers among patients that needed ICU support, mechanical ventilation or who had died, and a higher percentage of smokers among the severe cases(12). However, from their published data we can calculate that the smokers were 1.4 times more likely (RR=1.4, 95% CI: 0.98–2.00) to have severe symptoms of COVID-19 and approximately 2.4 times more likely to be admitted to an ICU, need mechanical ventilation or die compared to non-smokers (RR=2.4, 95% CI: 1.43–4.04). In conclusion, although further research is warranted as the weight of the evidence increases, with the limited available data, and although the above results are unadjusted for other factors that may impact disease progression, smoking is most likely associated with the negative progression and adverse outcomes of COVID-19."

def parse():
    model_dir = os.path.join('models', 'v' + version)
    nlp = spacy.load(model_dir)
    doc = nlp(text)

    print('*** Entities   ')
    for ent in doc.ents:
        print([ent.label_, ent.start_char, ent.end_char, ent.text])

    print('\n*** Tokens ***')
    for ent in doc:
        if ent.ent_iob_ != 'O':
            print([ent.text, ent.ent_iob_, ent.ent_type_])

if __name__ == "__main__":
    parse()