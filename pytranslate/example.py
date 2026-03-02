from pytranslate import AXTranslate
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--text', type=str)
    parser.add_argument('--target', type=str, default='Chinese')
    args = parser.parse_args()

    translate = AXTranslate(model_dir=args.model_dir)
    
    output = translate.translate(args.text, args.target)
    print(output)
    
    del translate
