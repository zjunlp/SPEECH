# speech-bert for trigger classification task on ontoevent-doc dataset
python3 run_speech.py \
    --data_dir ./Datasets/OntoEvent-Doc \
    --model_type speech_bert \
    --model_name_or_path bert-base-uncased \
    --task_name ontoevent-doc \
    --central_task token \
    --ere_task_type doc_all \
    --output_dir ./history_models/trigger-classification_ontoevent-doc_speech-bert \
    --max_seq_length 128 \
    --max_mention_size 50 \
    --do_lower_case \
    --per_gpu_train_batch_size 2 \
    --per_gpu_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --num_train_epochs 100 \
    --save_steps 500 \
    --logging_steps 500 \
    --seed 42 \
    --do_train \
    --do_eval \
    --do_test \
    --evaluate_during_training \
    --overwrite_output_dir

# speech-bert for event classification task on ontoevent-doc dataset
python3 run_speech.py \
    --data_dir ./Datasets/OntoEvent-Doc \
    --model_type speech_distilbert \
    --model_name_or_path distilbert-base-uncased \
    --task_name ontoevent-doc \
    --central_task sent \
    --ere_task_type doc_joint \
    --output_dir ./history_models/event-classification_ontoevent-doc_speech-distilbert  \
    --max_seq_length 128 \
    --max_mention_size 50 \
    --do_lower_case \
    --per_gpu_train_batch_size 1 \
    --per_gpu_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --num_train_epochs 100 \
    --save_steps 500 \
    --logging_steps 500 \
    --seed 42 \
    --do_train \
    --do_eval \
    --do_test \
    --evaluate_during_training \
    --overwrite_output_dir

# speech-distilbert for ERE (All Joint) task on ontoevent-doc dataset
python3 run_speech.py \
    --data_dir ./Datasets/OntoEvent-Doc \
    --model_type speech_distilbert \
    --model_name_or_path distilbert-base-uncased \
    --task_name ontoevent-doc \
    --central_task doc \
    --ere_task_type doc_all \
    --output_dir ./history_models/event-relation-classification_ontoevent-doc_speech-distilbert/all \
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
python3 run_speech.py \
    --data_dir ./Datasets/OntoEvent-Doc \
    --model_type speech_distilbert \
    --model_name_or_path distilbert-base-uncased \
    --task_name ontoevent-doc \
    --central_task doc \
    --ere_task_type doc_joint \
    --output_dir ./history_models/event-relation-classification_ontoevent-doc_speech-distilbert/joint \
    --max_seq_length 128 \
    --max_mention_size 50 \
    --do_lower_case \
    --per_gpu_train_batch_size 2 \
    --per_gpu_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --num_train_epochs 100 \
    --save_steps 500 \
    --logging_steps 500 \
    --seed 42 \
    --do_train \
    --do_eval \
    --do_test \
    --evaluate_during_training \
    --overwrite_output_dir

# speech-distilbert for ERE (Temporal) task on ontoevent-doc dataset
python3 run_speech.py \
    --data_dir ./Datasets/OntoEvent-Doc \
    --model_type speech_distilbert \
    --model_name_or_path distilbert-base-uncased \
    --task_name ontoevent-doc \
    --central_task doc \
    --ere_task_type doc_temporal \
    --output_dir ./history_models/event-relation-classification_ontoevent-doc_speech-distilbert/temporal \
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
python3 run_speech.py \
    --data_dir ./Datasets/OntoEvent-Doc \
    --model_type speech_distilbert \
    --model_name_or_path distilbert-base-uncased \
    --task_name ontoevent-doc \
    --central_task doc \
    --ere_task_type doc_causal \
    --output_dir ./history_models/event-relation-classification_ontoevent-doc_speech-distilbert/causal \
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
python3 run_speech.py \
    --data_dir ./Datasets/MAVEN_ERE \
    --model_type speech_bert \
    --model_name_or_path bert-base-uncased \
    --task_name maven-ere \
    --central_task token \
    --ere_task_type doc_all \
    --output_dir ./history_models/trigger-classification_maven-ere_speech-bert \
    --max_seq_length 128 \
    --max_mention_size 40 \
    --do_lower_case \
    --per_gpu_train_batch_size 1 \
    --per_gpu_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --num_train_epochs 100 \
    --save_steps 500 \
    --logging_steps 500 \
    --seed 42 \
    --do_train \
    --do_eval \
    --evaluate_during_training \
    --overwrite_output_dir

# speech-bert for event classification task on maven-ere dataset
python3 run_speech.py \
    --data_dir ./Datasets/MAVEN_ERE \
    --model_type speech_distilbert \
    --model_name_or_path distilbert-base-uncased \
    --task_name maven-ere \
    --central_task sent \
    --ere_task_type doc_joint \
    --output_dir ./history_models/event-classification_maven-ere_speech-distilbert  \
    --max_seq_length 128 \
    --max_mention_size 40 \
    --do_lower_case \
    --per_gpu_train_batch_size 1 \
    --per_gpu_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --num_train_epochs 100 \
    --save_steps 500 \
    --logging_steps 500 \
    --seed 42 \
    --do_train \
    --do_eval \
    --evaluate_during_training \
    --overwrite_output_dir

# speech-distilbert for ERE (All Joint) task on maven-ere dataset
python3 run_speech.py \
    --data_dir ./Datasets/MAVEN_ERE \
    --model_type speech_distilbert \
    --model_name_or_path distilbert-base-uncased \
    --task_name maven-ere \
    --central_task doc \
    --ere_task_type doc_all \
    --output_dir ./history_models/event-relation-classification_maven-ere_speech-distilbert/all \
    --max_seq_length 128 \
    --max_mention_size 40 \
    --do_lower_case \
    --per_gpu_train_batch_size 4 \
    --per_gpu_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --num_train_epochs 100 \
    --save_steps 500 \
    --logging_steps 500 \
    --seed 42 \
    --do_train \
    --do_eval \
    --evaluate_during_training \
    --overwrite_output_dir 

# speech-distilbert for ERE (Joint) task on maven-ere dataset
python3 run_speech.py \
    --data_dir ./Datasets/MAVEN_ERE \
    --model_type speech_distilbert \
    --model_name_or_path distilbert-base-uncased \
    --task_name maven-ere \
    --central_task doc \
    --ere_task_type doc_joint \
    --output_dir ./history_models/event-relation-classification_maven-ere_speech-distilbert/joint \
    --max_seq_length 128 \
    --max_mention_size 40 \
    --do_lower_case \
    --per_gpu_train_batch_size 4 \
    --per_gpu_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --num_train_epochs 100 \
    --save_steps 500 \
    --logging_steps 500 \
    --seed 42 \
    --do_train \
    --do_eval \
    --evaluate_during_training \
    --overwrite_output_dir

# speech-distilbert for ERE (Temporal) task on maven-ere dataset
python3 run_speech.py \
    --data_dir ./Datasets/MAVEN_ERE \
    --model_type speech_distilbert \
    --model_name_or_path distilbert-base-uncased \
    --task_name maven-ere \
    --central_task doc \
    --ere_task_type doc_temporal \
    --output_dir ./history_models/event-relation-classification_maven-ere_speech-distilbert/temporal \
    --max_seq_length 128 \
    --max_mention_size 40 \
    --do_lower_case \
    --per_gpu_train_batch_size 4 \
    --per_gpu_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --num_train_epochs 100 \
    --save_steps 500 \
    --logging_steps 500 \
    --seed 42 \
    --do_train \
    --do_eval \
    --evaluate_during_training \
    --overwrite_output_dir

# speech-distilbert for ERE (Causal) task on maven-ere dataset
python3 run_speech.py \
    --data_dir ./Datasets/MAVEN_ERE \
    --model_type speech_distilbert \
    --model_name_or_path distilbert-base-uncased \
    --task_name maven-ere \
    --central_task doc \
    --ere_task_type doc_causal \
    --output_dir ./history_models/event-relation-classification_maven-ere_speech-distilbert/causal \
    --max_seq_length 128 \
    --max_mention_size 40 \
    --do_lower_case \
    --per_gpu_train_batch_size 4 \
    --per_gpu_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --num_train_epochs 100 \
    --save_steps 500 \
    --logging_steps 500 \
    --seed 42 \
    --do_train \
    --do_eval \
    --evaluate_during_training \
    --overwrite_output_dir

# speech-distilbert for ERE (Subevent) task on maven-ere dataset
python3 run_speech.py \
    --data_dir ./Datasets/MAVEN_ERE \
    --model_type speech_distilbert \
    --model_name_or_path distilbert-base-uncased \
    --task_name maven-ere \
    --central_task doc \
    --ere_task_type doc_sub \
    --output_dir ./history_models/event-relation-classification_maven-ere_speech-distilbert/sub \
    --max_seq_length 128 \
    --max_mention_size 40 \
    --do_lower_case \
    --per_gpu_train_batch_size 1 \
    --per_gpu_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --num_train_epochs 100 \
    --save_steps 500 \
    --logging_steps 500 \
    --seed 42 \
    --do_train \
    --do_eval \
    --evaluate_during_training \
    --overwrite_output_dir

