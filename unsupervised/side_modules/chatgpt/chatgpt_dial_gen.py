import openai
import pandas as pd
import json
import argparse
import os

import re



def process_dial(dialogue, utter):
    d = []
    dialogue = dialogue.replace('\n\n', '\n')
    dialogue = dialogue.split('\n')
    
    utter = utter.replace('\n\n', '\n')
    utter = utter.split('\n')

    ppl = []
    for idx, dial in enumerate(dialogue):

        try:
            emotion = re.search('\[(.*?)\]', dial)
            emotion = emotion.group(1)
        except:
            emotion = 'none'
        
        dial = dial.replace('['+emotion+']', '')

        
        person, dial = dial.split(':')
        
        if '<TRUE>' in utter[idx]: 
                do_norm = 'True'
        else: do_norm = 'False'
        

        d_tmp = {}
        if person not in ppl: ppl.append(person)
        d_tmp['speaker'] = ppl.index(person)
        d_tmp['utterance'] = dial
        d_tmp['do_norm'] = do_norm
        d_tmp['emotion'] = emotion
        d.append(d_tmp)
    return d
        

    
norms = ['doing apology', 'doing criticism', 'doing greeting', 'doing request', \
         'doing persuasion', 'doing thanks','doing taking leave', 'doing admiration', \
         'doing refusing a request', 'doing finalising a negotiation', 'doing shaming', \
         'doing informing', 'doing forgiving', 'doing sympathizing', 'doing validating', \
         'doing expressing affection']
status = ['adhere to', 'violate']
language = ['English', 'Mandarin']



def main(args):
    
    openai.api_key = args.api_key
    
    
    with open(args.captions, 'r') as fp:
        captions = json.load(fp)
    
    if os.path.isfile(args.syn_dials):
        with open(args.syn_dials, 'r') as fp:
            syn_dials = json.load(fp)
    else:
        syn_dials = {}

    ctr = 0


    processed = {}

    for k, v in captions.items():
        k = k.split('_')[0]
        if k not in processed:
            for n in norms:
                for s in status:
                    for l in language:
                        processed[k] = 1
                    
                        ##dialogue
                        if l == 'English':
                            prompt = 'Generate a dialogue with up to 10 turns about the social situation of \'{}\'. \
                                  This dialogue should {} the socio-cultural norm of \'{}\'. \n For each line of \
                                  dialogue that you print, label the end of the line with the emotion of the speaker, \
                                  for example: \n "You are useless" [Frustrated]. '.format(v[0], s, n)
                        elif l == 'Mandarin':
                            prompt = 'Generate a dialogue in Mandarin with up to 10 turns about the social situation of \'{}\'. \
                                  This dialogue should {} the socio-cultural norm of \'{}\'. Do not generate the translation. \n \
                                  For each line of dialogue that you print, label the end of the line with the emotion of the speaker, \
                                  for example: \n "好的，谢谢你的帮助。" [Grateful]. '.format(v[0], s, n)
                        dic = {'role': 'user', 'content': prompt}
                        completion = openai.ChatCompletion.create(model=args.model_engine, messages = [dic])
                        dial = completion.choices[0].message['content']
                
                        ##utterence tag
                        prompt = 'Given the dialogue {}, which utterance {} to the socio-cultural norm of \'{}\'. \
                              For each line of the dialogue, label the end of the line with <TRUE> if the line \
                              contains the socio-cultural norm of \'{}\', otherwise label it with <FALSE>. \n'.format(dial.strip(), s, n, n)

                        dic = {'role': 'user', 'content': prompt}
                        completion = openai.ChatCompletion.create(model=args.model_engine, messages = [dic])
                        utter = completion.choices[0].message['content']
                
                        ##CoT
                        prompt = 'Generate a summarized \'Chain of Thoughts\' on why the dialogue {} {} to the socio-cultural norm of \'{}\'.\n'.format(dial.replace('\n\n', '\n'), s, n)
                        dic = {'role': 'user', 'content': prompt}
                        completion = openai.ChatCompletion.create(model=args.model_engine, messages = [dic])
                        cot = completion.choices[0].message['content']
                
                
                        try:
                            dial_processed = process_dial(dial.strip(), utter.strip())
                        except:
                            dial_processed = 'none'
                    
                
                        syn_dials[k + '_' + str(ctr)] = {}
                        syn_dials[k + '_' + str(ctr)]['social_situation'] = v[0]
                        syn_dials[k + '_' + str(ctr)]['norm'] = n
                        syn_dials[k + '_' + str(ctr)]['status'] = s
                        syn_dials[k + '_' + str(ctr)]['language'] = l
                        syn_dials[k + '_' + str(ctr)]['dialogue'] = dial
                        syn_dials[k + '_' + str(ctr)]['utterence_with_norm'] = utter
                        syn_dials[k + '_' + str(ctr)]['CoT'] = cot
                        syn_dials[k + '_' + str(ctr)]['processed_dialogue'] = dial_processed
                
                        with open('syn_dials.json', 'w') as fp: json.dump(syn_dials, fp)
                        with open('last_processed.txt', 'a+') as fp: fp.write(k + '\n')
                        ctr += 1
                        if ctr == 5000:
                            print('{} samples generated ...'.format(ctr))
                            exit(0)
        else:
            pass





if __name__ == '__main__':
         parser = argparse.ArgumentParser()
         parser.add_argument('--api_key', type=str, help = 'Your OpenAI api key')
         parser.add_argument('--model_engine', type=str, default='gpt-3.5-turbo')
         parser.add_argument('--syn_dials', type=str, default='syn_dials.json', help = 'save/load generated dialogues')
         parser.add_argument('--captions', type=str, default='video_text_Cap_sorted.json')

         args = parser.parse_args()

         opt = vars(args)

         main(args)

         print("done...")
