import tensorflow.keras as keras
import numpy as np
import json
import music21 as mu
class Melody_Generator:

    def __init__(self,model_path="C:/Users/user/PycharmProjects/Melody_Generation/new_dataset/lstm_model.h5"):
        self.model_path=model_path
        self.model=keras.models.load_model(model_path)


        with open("C:/Users/user/PycharmProjects/Melody_Generation/new_dataset/mapping","r") as fp:
            self.mapping=json.load(fp)

        self.start_symbol= ["/"] * 64


    def melody_generate(self, seed , num_steps,max_seq_length,temperature):
        self.seed=seed.split()
        melody=self.seed
        seed=self.start_symbol + self.seed
        print(seed)

        seed= [self.mapping[symbol] for symbol in seed]

        for p in range(num_steps):
            seed=seed[-max_seq_length:]


            encoded_seed= keras.utils.to_categorical(seed, num_classes=len(self.mapping))
            encoded_seed= encoded_seed[np.newaxis,...]

            probabilities_of_prediction= self.model.predict(encoded_seed)[0]
            output=self.decide_on_probabilities(probabilities_of_prediction,temperature)

            seed.append(output)

            output_symbol=[k for k,v in self.mapping.items() if v==output]


            if np.squeeze(output_symbol) == "/":

                break
            melody.append(str(np.squeeze(output_symbol)))

        return melody



    def decide_on_probabilities(self,probabilities,temperature):
        predictions=np.log(probabilities)/temperature
        probabilities=np.exp(predictions)/np.sum(np.exp(predictions))
        choice=np.random.choice(range(len(probabilities)),p=probabilities)

        return choice


    def save_melody(self,melody,format="midi",step_duration=0.25,file_name="generated_melody.midi"):
        stream=mu.stream.Stream()
        step_counter=1
        start=None
        event=None

        for i,symbol in enumerate(melody):
            if symbol !="-" :

                start = symbol
                if start is not None:
                    quarter_length=step_duration*step_counter

                    if start=='r':
                        event=mu.note.Rest(quarter_length=quarter_length)

                    else:


                        event=mu.note.Note(int(symbol),quarter_length=quarter_length)
                if event is not None:
                    stream.append(event)
                step_counter=1

            else:
                step_counter=step_counter+1
        stream.write(format,file_name)
mg_1=Melody_Generator()
seed="76 - - - - - 74 - 72 - 64 - 69 - - - 67 - 65 - 62 - - - 69 - - - 67 - 65 - 62 - - - 69 "
melody=mg_1.melody_generate(seed,500,64,0.7)
print(melody)
mg_1.save_melody(melody)
