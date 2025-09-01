import onnxruntime as rt
import numpy as np
from transformers import AutoTokenizer
from typing import List 


def run_axegine(sentences: List[str]):
    import axengine as axrt

    session = axrt.InferenceSession("compiled.axmodel")
    
    for input in session.get_inputs():
        print("Input Name:", input.name, ", Shape:", input.shape, ", Type:", input.dtype)

    for output in session.get_outputs():
        print("Output Name:", output.name, ", Shape:", output.shape, ", Type:", output.dtype)
    # Load the tokenizer

    # Load the tokenizer
    max_length = 77
    model_path = "opus-mt-en-zh/"
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    result = []
    for st in sentences:
        encoded_input = tokenizer(st, return_tensors="pt", padding='max_length', max_length=max_length,
                                truncation=True)

        # Prepare inputs for the model
        input_ids = np.array(encoded_input.input_ids).astype(np.int32)
        attention_mask = np.array(encoded_input.attention_mask).astype(np.int32)

        # Initialize decoder input ids with a valid start token

        start_token_id = tokenizer.pad_token_id
        decoder_input_ids = np.full((1, 1), start_token_id).astype(np.int32).repeat(77,1)

        # Initialize decoder attention mask
        decoder_attention_mask = np.zeros((1, 1)).astype(np.int32).repeat(77,1)
        decoder_attention_mask[0][0]=1
        # Maximum length for the generated sequence
        max_decoder_length = max_length
        
        # print(input_ids)
        # print(attention_mask)
        print(decoder_input_ids)
        print(decoder_attention_mask)
        

        # Generate the sequence
        for idx in range(max_decoder_length - 1):  # -1 because we already have the start token
            model_input = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'decoder_input_ids': decoder_input_ids,
                'decoder_attention_mask': decoder_attention_mask,
            }
            
            predicted_token_id = session.run(None, model_input)[0]
            decoder_input_ids[0][idx+1:] = predicted_token_id
            decoder_attention_mask[0][idx+1] = 1
            
            # print(decoder_input_ids)
            # print(decoder_attention_mask)


        predicted_sequence = decoder_input_ids[0]
        decoded_output = tokenizer.decode(predicted_sequence, skip_special_tokens=True)
        result.append(decoded_output)
        
    # Print the result
    return result



if __name__ == "__main__":    
    sentences = [
        "A man is riding a red motorcycle", 
        # "A shirtless person",  
        # "A woman riding a motorcycle with a helmet on",  
        # "An elderly woman wearing a brown hat and carrying a crossbody bag.",  
        # "A person carrying a purple suitcase",  
        # "There is someone on the back of the motorcycle",  
        # "A short-haired woman walking a dog.",  
        # "Grey checkered pattern top, wearing a hat for the old lady.",  
        # "A white tricycle",  
        # "A man wearing a brown top and blue jeans",  
        # "A woman wearing a gray top and black pants",  
        # "A woman wearing a gray short-sleeve top and black high heels",  
        # "A woman wearing a white top and a brown skirt",  
        # "A young woman wearing a white top and brown shorts.",  
        # "A young woman wearing a white top, purple shorts, and carrying a backpack.",  
        # "A woman wearing a white top and blue jeans",  
        # "A woman wearing a white top and black shorts",  
        # "A woman wearing a white top and black pants",  
        # "A man wearing a brown shirt, glasses, and a mask.",  
        # "An elderly woman wearing a light blue long-sleeve top, white sneakers, and a hat",  
        # "A woman wearing a white top, black pants, carrying a black backpack, and holding an umbrella.",  
        # "A woman wearing a white down jacket and carrying a black bag",  
        # "A woman wearing a white down jacket and carrying a shoulder bag.",  
        # "A woman wearing a white down jacket and carrying a backpack.",  
        # "A man wearing a purple top.",  
        # "A middle-aged woman wearing a red top and black pants.",  
        # "A sanitation worker wearing a green reflective vest",  
        # "A middle-aged man wearing a blue top and glasses.",  
        # "A bald man wearing a yellow and white striped top",  
        # "A middle-aged man wearing a black top, brown shorts, and glasses.",  
        # "A man wearing a black top riding a yellow bicycle.",  
        # "A young woman wearing a black top, black pants, a mask, and carrying a bag.",  
        # "A woman wearing a purple top and black pants.",  
        # "A woman wearing a red top and black pants.",  
        # "An elderly person wearing a green top, a purple hat, and carrying a backpack.",  
        # "An elderly person wearing a green top, a yellow hat, and carrying a blue bag.",  
        # "An elderly person wearing a green top, a black hat, and carrying a backpack.",  
        # "A child wearing a green vest.",  
        # "A man wearing Adidas clothing.",  
        # "A woman wearing a black top and blue jeans.",  
        # "A man wearing a black top and black pants.",  
        # "A woman with red hair",  
        # "The man carrying a black backpack",  
        # "A man with blue short sleeves and white long pants.",  
        # "A woman with yellow hair.",  
        # "A man with yellow hair",  
        # "An elderly woman wearing a yellow hat and black pants.",  
        # "Yellow patterned top, Grandma wearing a hat.",  
        # "Facial features with long black hair",  
    ]
    res_ort = run_axegine(sentences)
    print(res_ort)


    # res_ort = run_ort(sentences)
    # res_ort1 = run_ort1(sentences)
    # for o, r in zip(res_ort, res_ort1):
    #     print("ort vs ppq :",  o, " | " , r)