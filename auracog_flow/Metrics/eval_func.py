import os
import mlflow
import json

# Extracting Reports
# ******************************

def extract_report(result_file, filename):

    doc_report = open(filename, 'a')

    lista_ent = []
    linea_ent_ind = -1
    words_ent = ""
    word_ent_ind = -1

    for i in result_file:
        word_ent_ind = word_ent_ind + 1

        if (i == " "):
            continue
        elif (i == "\n"):
            linea_ent_ind = linea_ent_ind + 1
            if linea_ent_ind == 0:
                words_ent = ',' + words_ent
            lista_ent.append(words_ent)
            doc_report.write(words_ent + '\n')
            words_ent = ""
        else:
            words_ent += str(i)
            if (result_file[word_ent_ind + 1]) == " ":
                words_ent += str(',')

    doc_report.close()
    mlflow.log_artifact(filename, 'reports')
    os.remove (filename)

# Extracting Metrics
# ******************************

def extract_metrics(result):

    with open('datos.json', 'w') as file:
        json.dump(result, file)

    mlflow.log_artifact('datos.json', 'results')

    with open('datos.json', 'r') as file:
        y = json.load(file)

        precision_int = y['intent_evaluation']['precision']
        f1_score_int = y['intent_evaluation']['f1_score']
        accuracy_int = y['intent_evaluation']['accuracy']

        mlflow.log_metric('precision_int', precision_int)
        mlflow.log_metric('f1_score_int', f1_score_int)
        mlflow.log_metric('accuracy_int', accuracy_int)

        precision_ent = y['entity_evaluation']['ner_crf']['precision']
        f1_score_ent = y['entity_evaluation']['ner_crf']['f1_score']
        accuracy_ent = y['entity_evaluation']['ner_crf']['accuracy']

        mlflow.log_metric('precision_ent', precision_ent)
        mlflow.log_metric('f1_score_ent', f1_score_ent)
        mlflow.log_metric('accuracy_ent', accuracy_ent)

    os.remove('datos.json')


#     Logging Model
#************************************

def keep_model(path):

    for i in os.listdir(path):

        if os.path.isdir(path + '/' + i):
            keep_model(path + '/' + i)

        else:
            mlflow.log_artifact(path + '/' + i, path)
