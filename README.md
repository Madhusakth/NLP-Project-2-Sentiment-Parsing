# NLP-Project-2-Sentiment-Parsing
Semantic Parsing with Encoder-Decoder Models
#attention model
python3 main_attention.py 

#attention with scheduled sampling
python3 main_scheduled.py
Please refer to this readme to understand the naming conventions used. 
geo_test_output_enc_dec_512.tsv: This is the simple encoder-decoder model with 512 hidden units and teacher forcing ratio set at 0.5. 

geo_test_output_enc_dec_512_linear.tsv: encoder-decoder model with 512 hidden units and linear decay scheduled sampling. 

geo_test_output_enc_dec_512_invsig.tsv: encoder-decoder model with 512 hidden units and inverse sigmoid decay scheduled sampling. 

geo_test_output_enc_dec_512_expo.tsv: encoder-decoder model with 512 hidden units and exponential decay scheduled sampling. 

geo_test_output_enc_dec_atten.tsv: Attention based encoder-decoder model with 512 hidden units and teacher forcing ratio set at 0.5.

geo_test_output_enc_dec_atten_invsig.tsv: Attention based encoder-decoder model with 512 hidden units and inverse sigmoid decay scheduled sampling. 

geo_test_output_enc_dec_atten_expo.tsv: Attention based encoder-decoder model with 512 hidden units and exponential decay scheduled sampling. 

geo_test_output_enc_dec_atten_linear.tsv: Attention based encoder-decoder model with 512 hidden units and linear decay scheduled sampling. 

I have also included the training epochs and their result. The denotation is calculated only on the dev set and dev accuracy is calculated every alternate epoch while training. 
