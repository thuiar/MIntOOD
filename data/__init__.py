

benchmarks = {
    'MIntRec':{
        'intent_labels': [
                    'Complain', 'Praise', 'Apologise', 'Thank', 'Criticize', 
                    'Agree', 'Taunt', 'Flaunt', 
                    'Joke', 'Oppose', 
                    'Comfort', 'Care', 'Inform', 'Advise', 'Arrange', 'Introduce', 'Leave', 
                    'Prevent', 'Greet', 'Ask for help' 
        ],
        'binary_maps': {
                    'Complain': 'Emotion', 'Praise':'Emotion', 'Apologise': 'Emotion', 'Thank':'Emotion', 'Criticize': 'Emotion',
                    'Care': 'Emotion', 'Agree': 'Emotion', 'Taunt': 'Emotion', 'Flaunt': 'Emotion',
                    'Joke':'Emotion', 'Oppose': 'Emotion', 
                    'Inform':'Goal', 'Advise':'Goal', 'Arrange': 'Goal', 'Introduce': 'Goal', 'Leave':'Goal',
                    'Prevent':'Goal', 'Greet': 'Goal', 'Ask for help': 'Goal', 'Comfort': 'Goal'
        },
        'binary_intent_labels': ['Emotion', 'Goal'],
        'label_len': 4,
        'max_seq_lengths': {
            'text': 30,
            'video': 230, 
            'audio': 480
        },
        'feat_dims': {
            'text': 768,
            'video': 1024,
            'audio': 768
        },
        'ood_data':{
            'MIntRec-OOD': {'ood_label': 'UNK'},
            'TED-OOD': {'ood_label': 'UNK'},
            'IEMOCAP-DA': {'ood_label': 'oth'},
            'IEMOCAP-DA-OOD': {'ood_label': 'oth'},
            'MELD-DA': {'ood_label': 'oth'},
            'MELD-DA-OOD': {'ood_label': 'oth'}
        }
    },
    'MIntRec2.0':{
        'intent_labels': [
            'Acknowledge', 'Advise', 'Agree', 'Apologise', 'Arrange', 
            'Ask for help', 'Asking for opinions', 'Care', 'Comfort', 'Complain', 
            'Confirm', 'Criticize', 'Doubt', 'Emphasize', 'Explain', 
            'Flaunt', 'Greet', 'Inform', 'Introduce', 'Invite', 
            'Joke', 'Leave', 'Oppose', 'Plan', 'Praise', 
            'Prevent', 'Refuse', 'Taunt', 'Thank', 'Warn',
        ],
        'max_seq_lengths': {
            'text': 50, # truth: 51 (max), 23 (mean+3std)
            'video': 180, # truth: 475 (max), 67 (avg), 181 (mean+3std)
            'audio':  400, # truth: 992 (max), 387 (mean+3std),
        },
        'feat_dims': {
            'text': 1024,
            'video': 256,
            'audio': 768
        },
        'ood_data': {
            'MIntRec2.0-OOD': {'ood_label': 'UNK'}
        }
    },
    'MELD-DA':{
        'intent_labels': [
                    'Greeting', 'Question', 'Answer', 'Statement Opinion', 'Statement Non Opinion', 
                    'Apology', 'Command', 'Agreement', 'Disagreement', 
                    'Acknowledge', 'Backchannel'
        ],
        'label_maps': {
                    'g': 'Greeting', 'q': 'Question', 'ans': 'Answer', 'o': 'Statement Opinion', 's': 'Statement Non Opinion', 
                    'ap': 'Apology', 'c': 'Command', 'ag': 'Agreement', 'dag': 'Disagreement', 
                    'a': 'Acknowledge', 'b': 'Backchannel'
        },
        'max_seq_lengths': {
            'text': 70, # max: 69, final: 27
            'video': 250, ### max: 618, final: 242
            'audio': 530 ### max 2052, final: 524
        },
        'label_len': 3,
        'feat_dims': {
            'text': 768,
            'video': 1024,
            'audio': 768
        },
        'ood_data':{
            'MELD-DA-OOD': {'ood_label': 'oth'},
            'MIntRec-OOD': {'ood_label': 'UNK'},
            'TED-OOD': {'ood_label': 'UNK'},
            'IEMOCAP-DA': {'ood_label': 'oth'},
            'IEMOCAP-DA-OOD': {'ood_label': 'oth'},
            'MIntRec': {'ood_label': 'UNK'}
        }
    },
    'IEMOCAP-DA':{
        'intent_labels': [
                    'Greeting', 'Question', 'Answer', 'Statement Opinion', 'Statement Non Opinion', 
                    'Apology', 'Command', 'Agreement', 'Disagreement', 
                    'Acknowledge', 'Backchannel'
        ],
        'label_maps': {
                    'g': 'Greeting', 'q': 'Question', 'ans': 'Answer', 'o': 'Statement Opinion', 's': 'Statement Non Opinion', 
                    'ap': 'Apology', 'c': 'Command', 'ag': 'Agreement', 'dag': 'Disagreement', 
                    'a': 'Acknowledge', 'b': 'Backchannel'
        },
        'max_seq_lengths': {
                'text': 44,
                'video': 230, # mean+sigma 
                'audio': 380
        },
        'label_len': 3,
        'feat_dims': {
            'text': 768,
            'video': 1024,
            'audio': 768
        },
        'ood_data':{
            'IEMOCAP-DA-OOD': {'ood_label': 'oth'},
            'MELD-DA': {'ood_label': 'oth'},
            'MELD-DA-OOD': {'ood_label': 'oth'},
            'MIntRec-OOD': {'ood_label': 'UNK'},
            'TED-OOD': {'ood_label': 'UNK'},
            'MIntRec': {'ood_label': 'UNK'},
        }
    },
}

