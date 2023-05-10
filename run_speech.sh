# speech-bert for trigger classification task on ontoevent-doc dataset
# please check ```speech_distilbert.py``` file first, where line 26-28 should be commented and line 29-31 should be uncommented 
python3 run_speech.py \
    --data_dir ./Datasets/OntoEvent-Doc \
    --model_type speech_bert \
    --model_name_or_path bert-base-uncased \
    --task_name ontoevent-doc \
    --central_task token \
    --ere_task_type doc_all \
    --output_dir ./history_models/speech-bert_ontoevent_trigger-classification \
    --max_seq_length 128 \
    --max_mention_size 50 \
    --do_lower_case \
    --per_gpu_train_batch_size 2 \
    --per_gpu_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --num_train_epochs 40 \
    --save_steps 500 \
    --logging_steps 500 \
    --seed 42 \
    --do_train \
    --do_eval \
    --do_test \
    --evaluate_during_training \
    --overwrite_output_dir

# speech-bert for event classification task on ontoevent-doc dataset
# please check ```speech_distilbert.py``` file first, where line 26-28 should be commented and line 29-31 should be uncommented 
python3 run_speech.py \
    --data_dir ./Datasets/OntoEvent-Doc \
    --model_type speech_bert \
    --model_name_or_path bert-base-uncased \
    --task_name ontoevent-doc \
    --central_task sent \
    --ere_task_type doc_joint \
    --output_dir ./history_models/speech-bert_ontoevent_event-classification \
    --max_seq_length 128 \
    --max_mention_size 50 \
    --do_lower_case \
    --per_gpu_train_batch_size 1 \
    --per_gpu_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --num_train_epochs 40 \
    --save_steps 500 \
    --logging_steps 500 \
    --seed 42 \
    --do_train \
    --do_eval \
    --do_test \
    --evaluate_during_training \
    --overwrite_output_dir

# speech-distilbert for ERE (All Joint) task on ontoevent-doc dataset
# please check ```speech_distilbert.py``` file first, where line 26-28 should be uncommented and line 29-31 should be commented 
python3 run_speech.py \
    --data_dir ./Datasets/OntoEvent-Doc \
    --model_type speech_distilbert \
    --model_name_or_path distilbert-base-uncased \
    --task_name ontoevent-doc \
    --central_task doc \
    --ere_task_type doc_all \
    --output_dir ./history_models/speech-distilbert_ontoevent_ere-all \
    --max_seq_length 128 \
    --max_mention_size 50 \
    --do_lower_case \
    --per_gpu_train_batch_size 2 \
    --per_gpu_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --num_train_epochs 40 \
    --save_steps 500 \
    --logging_steps 500 \
    --seed 42 \
    --do_train \
    --do_eval \
    --do_test \
    --evaluate_during_training \
    --overwrite_output_dir 

# speech-distilbert for ERE (Joint) task on ontoevent-doc dataset
# please check ```speech_distilbert.py``` file first, where line 26-28 should be uncommented and line 29-31 should be commented 
python3 run_speech.py \
    --data_dir ./Datasets/OntoEvent-Doc \
    --model_type speech_distilbert \
    --model_name_or_path distilbert-base-uncased \
    --task_name ontoevent-doc \
    --central_task doc \
    --ere_task_type doc_joint \
    --output_dir ./history_models/speech-distilbert_ontoevent_ere-joint \
    --max_seq_length 128 \
    --max_mention_size 50 \
    --do_lower_case \
    --per_gpu_train_batch_size 2 \
    --per_gpu_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --num_train_epochs 40 \
    --save_steps 500 \
    --logging_steps 500 \
    --seed 42 \
    --do_train \
    --do_eval \
    --do_test \
    --evaluate_during_training \
    --overwrite_output_dir

# speech-distilbert for ERE (Temporal) task on ontoevent-doc dataset
# please check ```speech_distilbert.py``` file first, where line 26-28 should be uncommented and line 29-31 should be commented 
python3 run_speech.py \
    --data_dir ./Datasets/OntoEvent-Doc \
    --model_type speech_distilbert \
    --model_name_or_path distilbert-base-uncased \
    --task_name ontoevent-doc \
    --central_task doc \
    --ere_task_type doc_temporal \
    --output_dir ./history_models/speech-distilbert_ontoevent_ere-temporal \
    --max_seq_length 128 \
    --max_mention_size 50 \
    --do_lower_case \
    --per_gpu_train_batch_size 2 \
    --per_gpu_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --num_train_epochs 40 \
    --save_steps 500 \
    --logging_steps 500 \
    --seed 42 \
    --do_train \
    --do_eval \
    --do_test \
    --evaluate_during_training \
    --overwrite_output_dir

# speech-distilbert for ERE (Causal) task on ontoevent-doc dataset
# please check ```speech_distilbert.py``` file first, where line 26-28 should be commented and line 29-31 should be uncommented 
python3 run_speech.py \
    --data_dir ./Datasets/OntoEvent-Doc \
    --model_type speech_distilbert \
    --model_name_or_path distilbert-base-uncased \
    --task_name ontoevent-doc \
    --central_task doc \
    --ere_task_type doc_causal \
    --output_dir ./history_models/speech-distilbert_ontoevent_ere-causal \
    --max_seq_length 128 \
    --max_mention_size 50 \
    --do_lower_case \
    --per_gpu_train_batch_size 2 \
    --per_gpu_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --num_train_epochs 40 \
    --save_steps 500 \
    --logging_steps 500 \
    --seed 42 \
    --do_train \
    --do_eval \
    --do_test \
    --evaluate_during_training \
    --overwrite_output_dir


# speech-bert for trigger classification task on maven-ere dataset
# please check ```speech_distilbert.py``` file first, where line 26-28 should be commented and line 29-31 should be uncommented 
python3 run_speech.py \
    --data_dir ./Datasets/MAVEN_ERE \
    --model_type speech_bert \
    --model_name_or_path bert-base-uncased \
    --task_name maven-ere \
    --central_task token \
    --ere_task_type doc_all \
    --output_dir ./history_models/speech-bert_maven_trigger-classification \
    --max_seq_length 128 \
    --max_mention_size 40 \
    --do_lower_case \
    --per_gpu_train_batch_size 2 \
    --per_gpu_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --num_train_epochs 40 \
    --save_steps 500 \
    --logging_steps 500 \
    --seed 42 \
    --do_train \
    --do_eval \
    --do_test \
    --evaluate_during_training \
    --overwrite_output_dir

# speech-bert for event classification task on maven-ere dataset
# please check ```speech_distilbert.py``` file first, where line 26-28 should be commented and line 29-31 should be uncommented 
python3 run_speech.py \
    --data_dir ./Datasets/MAVEN_ERE \
    --model_type speech_bert \
    --model_name_or_path bert-base-uncased \
    --task_name maven-ere \
    --central_task sent \
    --ere_task_type doc_joint \
    --output_dir ./history_models/speech-bert_maven_event-classification \
    --max_seq_length 128 \
    --max_mention_size 40 \
    --do_lower_case \
    --per_gpu_train_batch_size 1 \
    --per_gpu_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --num_train_epochs 40 \
    --save_steps 500 \
    --logging_steps 500 \
    --seed 42 \
    --do_train \
    --do_eval \
    --do_test \
    --evaluate_during_training \
    --overwrite_output_dir

# speech-distilbert for ERE (All Joint) task on maven-ere dataset
# please check ```speech_distilbert.py``` file first, where line 26-28 should be uncommented and line 29-31 should be commented 
python3 run_speech.py \
    --data_dir ./Datasets/MAVEN_ERE \
    --model_type speech_distilbert \
    --model_name_or_path distilbert-base-uncased \
    --task_name maven-ere \
    --central_task doc \
    --ere_task_type doc_all \
    --output_dir ./history_models/speech-distilbert_maven_ere-all \
    --max_seq_length 128 \
    --max_mention_size 40 \
    --do_lower_case \
    --per_gpu_train_batch_size 2 \
    --per_gpu_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --num_train_epochs 40 \
    --save_steps 500 \
    --logging_steps 500 \
    --seed 42 \
    --do_train \
    --do_eval \
    --do_test \
    --evaluate_during_training \
    --overwrite_output_dir 

# speech-distilbert for ERE (Joint) task on maven-ere dataset
# please check ```speech_distilbert.py``` file first, where line 26-28 should be uncommented and line 29-31 should be commented 
python3 run_speech.py \
    --data_dir ./Datasets/MAVEN_ERE \
    --model_type speech_distilbert \
    --model_name_or_path distilbert-base-uncased \
    --task_name maven-ere \
    --central_task doc \
    --ere_task_type doc_joint \
    --output_dir ./history_models/speech-distilbert_maven_ere-joint \
    --max_seq_length 128 \
    --max_mention_size 40 \
    --do_lower_case \
    --per_gpu_train_batch_size 2 \
    --per_gpu_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --num_train_epochs 40 \
    --save_steps 500 \
    --logging_steps 500 \
    --seed 42 \
    --do_train \
    --do_eval \
    --do_test \
    --evaluate_during_training \
    --overwrite_output_dir

# speech-distilbert for ERE (Temporal) task on maven-ere dataset
# please check ```speech_distilbert.py``` file first, where line 26-28 should be uncommented and line 29-31 should be commented 
python3 run_speech.py \
    --data_dir ./Datasets/MAVEN_ERE \
    --model_type speech_distilbert \
    --model_name_or_path distilbert-base-uncased \
    --task_name maven-ere \
    --central_task doc \
    --ere_task_type doc_temporal \
    --output_dir ./history_models/speech-distilbert_maven_ere-temporal \
    --max_seq_length 128 \
    --max_mention_size 40 \
    --do_lower_case \
    --per_gpu_train_batch_size 2 \
    --per_gpu_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --num_train_epochs 40 \
    --save_steps 500 \
    --logging_steps 500 \
    --seed 42 \
    --do_train \
    --do_eval \
    --do_test \
    --evaluate_during_training \
    --overwrite_output_dir

# speech-distilbert for ERE (Causal) task on maven-ere dataset
# please check ```speech_distilbert.py``` file first, where line 26-28 should be commented and line 29-31 should be uncommented 
python3 run_speech.py \
    --data_dir ./Datasets/MAVEN_ERE \
    --model_type speech_distilbert \
    --model_name_or_path distilbert-base-uncased \
    --task_name maven-ere \
    --central_task doc \
    --ere_task_type doc_causal \
    --output_dir ./history_models/speech-distilbert_maven_ere-causal \
    --max_seq_length 128 \
    --max_mention_size 40 \
    --do_lower_case \
    --per_gpu_train_batch_size 2 \
    --per_gpu_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --num_train_epochs 40 \
    --save_steps 500 \
    --logging_steps 500 \
    --seed 42 \
    --do_train \
    --do_eval \
    --do_test \
    --evaluate_during_training \
    --overwrite_output_dir

# speech-distilbert for ERE (Subevent) task on maven-ere dataset
# please check ```speech_distilbert.py``` file first, where line 26-28 should be uncommented and line 29-31 should be commented 
python3 run_speech.py \
    --data_dir ./Datasets/MAVEN_ERE \
    --model_type speech_distilbert \
    --model_name_or_path distilbert-base-uncased \
    --task_name maven-ere \
    --central_task doc \
    --ere_task_type doc_sub \
    --output_dir ./history_models/speech-distilbert_maven_ere-sub \
    --max_seq_length 128 \
    --max_mention_size 40 \
    --do_lower_case \
    --per_gpu_train_batch_size 2 \
    --per_gpu_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --num_train_epochs 40 \
    --save_steps 500 \
    --logging_steps 500 \
    --seed 42 \
    --do_train \
    --do_eval \
    --do_test \
    --evaluate_during_training \
    --overwrite_output_dir

