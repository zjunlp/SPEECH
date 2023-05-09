# MAVEN-ERE & OntoEvent-Doc

🍎 This is a repository for [**OntoEvent-Doc**](https://github.com/zjunlp/SPEECH/tree/main/Datasets/OntoEvent-Doc.zip) dataset.

💡 Note that [**MAVEN_ERE**](https://github.com/zjunlp/SPEECH/tree/main/Datasets/MAVEN_ERE.zip) is proposed in a [paper](https://aclanthology.org/2022.emnlp-main.60) and released in [GitHub](https://github.com/THU-KEG/MAVEN-ERE), where introduced the detailed data schema.  

## Data File Structure
The structure of data files (require to unzip [**MAVEN_ERE**](https://github.com/zjunlp/SPEECH/tree/main/Datasets/MAVEN_ERE.zip) and [**OntoEvent-Doc**](https://github.com/zjunlp/SPEECH/tree/main/Datasets/OntoEvent-Doc.zip) first) is as follows: 

```shell
SPEECH
└── Datasets
    ├── MAVEN_ERE   
    │   ├── train.jsonl     # for training
    │   ├── test.jsonl      # for testing
    │   └── valid.jsonl     # for validation
    └── OntoEvent-Doc 
    │   ├── event_dict_on_doc_train.json	# for training
    │   ├── event_dict_on_doc_test.json		# for testing
    │   └── event_dict_on_doc_valid.json	# for validation
    └── README.md 
```

## Brief Introduction
[**OntoEvent-Doc**](https://github.com/zjunlp/SPEECH/tree/main/Datasets/OntoEvent-Doc.zip), formatted in document level, is derived from [OntoEvent](https://github.com/231sm/Reasoning_In_EE/tree/main/OntoEvent) which is formatted in sentence level.  


## Statistics
The statistics of *MAVEN-ERE* and *OntoEvent-Doc* are shown below. 

Dataset         | #Document | #Mention | #Temporal | #Causal | #Subevent |
| :----------------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- |
MAVEN-ERE        | 4,480 | 112,276 | 1,216,217 | 57,992  | 15,841 |
OntoEvent-Doc    | 4,115 | 60,546 | 5,914 | 14,155 | / |


## Data Format
The data schema of MAVEN-ERE can be referred to their [GitHub](https://github.com/THU-KEG/MAVEN-ERE). 

The OntoEvent-Doc dataset is stored in json format.

🍒Each *document* (specialized with a *doc_id*, e.g., 95dd35ce7dd6d377c963447eef47c66c) in OntoEvent-Doc datasets contains a list of "events" and a dictionary of "relations", where the data format is as below:

```
[a doc_id]:
{
    "events": [
    {
        'doc_id': '...', 
        'doc_title': 'XXX', 
        'sent_id': , 
        'event_mention': '......', 
        'event_mention_tokens': ['.', '.', '.', '.', '.', '.'], 
        'trigger': '...', 
        'trigger_pos': [, ], 
        'event_type': ''
    },
    {
        'doc_id': '...', 
        'doc_title': 'XXX', 
        'sent_id': , 
        'event_mention': '......', 
        'event_mention_tokens': ['.', '.', '.', '.', '.', '.'], 
        'trigger': '...', 
        'trigger_pos': [, ], 
        'event_type': ''
    },
    ... 
    ],
    "relations": { // each event-relation contains a list of 'sent_id' pairs.  
        "COSUPER": [[,], [,], [,]], 
        "SUBSUPER": [], 
        "SUPERSUB": [], 
        "CAUSE": [[,], [,]], 
        "BEFORE": [[,], [,]], 
        "AFTER": [[,], [,]], 
        "CAUSEDBY": [[,], [,]], 
        "EQUAL": [[,], [,]]
    }
} 
```
