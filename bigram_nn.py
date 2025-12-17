import torch
import torch.nn.functional as F

file_path = "names.txt"
split_ratio = 0.9

class BigramNN():

    def __init__(self,file_path,split_ratio):
        self.file_path = file_path
        self.split_ratio = split_ratio
        self.lr = 0.1
        self.read_file()
        
    def read_file(self):
        with open(self.file_path,'r') as f:
            self.words = f.read().split()
        self.data_preprocessing()
        
    def data_preprocessing(self):
        self.chars = list(set("".join(self.words)))
        self.chars += ["<S>","<E>"]
        self.vocab_size = len(self.chars)
        self.strTOint()
        self.intTOstr()
    
    def strTOint(self):
        self.stoi = {}
        for i,ch in enumerate(self.chars):
            self.stoi[ch] = i

    def intTOstr(self):
        self.itos = {}
        for i,ch in enumerate(self.chars):
            self.itos[i] = ch
        self.train_test_split()
        
    def train_test_split(self):
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []
        
        n = int(self.split_ratio*len(self.words))
        self.train = self.words[:n]
        self.test = self.words[n:]
        
        for w in self.train:
            w = ["<S>"] + list(w) + ["<E>"]
            for i,ch in enumerate(w[:-1]):
                self.x_train.append(self.stoi[ch])
                self.y_train.append(self.stoi[w[i+1]])
        
        for w in self.test:
            w = ["<S>"] + list(w) + ["<E>"]
            for i,ch in enumerate(w[:-1]):
                self.x_test.append(self.stoi[ch])
                self.y_test.append(self.stoi[w[i+1]])
        
        self.x_train_tensor = torch.tensor(self.x_train)
        self.y_train_tensor = torch.tensor(self.y_train)
        self.x_test_tensor = torch.tensor(self.x_test)
        self.y_test_tensor = torch.tensor(self.y_test)
        print(self.x_train_tensor.shape,self.y_train_tensor.shape,self.x_test_tensor.shape,self.y_test_tensor.shape)
        self.one_hot_encodinng()

    def one_hot_encodinng(self):
        self.x_train_enc = F.one_hot(self.x_train_tensor, num_classes = self.vocab_size).float()
        self.x_test_enc = F.one_hot(self.x_test_tensor, num_classes = self.vocab_size).float()
        self.neural_network()

    def neural_network(self):
        self.hidden_neurons = 2*self.vocab_size
        self.epochs = 500
        self.W = torch.randn((self.vocab_size,self.vocab_size), requires_grad = True)
        self.b = torch.randn((self.vocab_size), requires_grad = True)

        for self.epoch in range(self.epochs):
            
            self.logits = self.x_train_enc @ self.W + self.b
            self.loss = F.cross_entropy(self.logits,self.y_train_tensor)
            self.loss.backward()
            if self.epoch % 100 == 0:
                print(f'At Epoch {self.epoch}, the loss is {self.loss.item():.3f}')
        
            with torch.no_grad():
                self.W -= self.lr * self.W.grad
                self.b -= self.lr * self.b.grad
            
            self.W.grad.zero_()
            self.b.grad.zero_()
            
        self.evaluation()

    def evaluation(self):
        
        # Evaluation
        self.logits_test = self.x_test_enc @ self.W + self.b
        self.loss_test = F.cross_entropy(self.logits_test,self.y_test_tensor)
        print(f'Test Loss = {self.loss_test:.3f}')
        
        # self.preds_test = torch.argmax(self.logits_test,dim=1)
        # acc = 0
        # for ind,pred in enumerate(self.preds_test):
        #     if pred.item() == self.y_test_tensor[ind]:
        #         acc += 1
        # acc = acc / self.preds_test.shape[0]
        # print(f'Accuracy = {acc:.2f}')
        self.name_generation()

    def name_generation(self):
        g = torch.Generator()
        self.input_seq = ["<S>"]
        self.input_seq_len = 0
        
        while self.input_seq_len <= 10:
            ind = self.stoi[self.input_seq[-1]]
            ind = torch.tensor(ind)
            enc_ind = F.one_hot(ind,num_classes = self.vocab_size).float()
            logits_lay = enc_ind @ self.W + self.b
            print("Logits Lay Shape = ",logits_lay.shape)
            probs = F.softmax(logits_lay, dim=-1)
            ind = torch.multinomial(probs, num_samples = 1, generator = g)
            nxt_ch = self.itos[ind.item()]
            self.input_seq.append(nxt_ch)
            if nxt_ch == "<E>":
                break
            self.input_seq_len +=1
        print("Name = ","".join(self.input_seq))

b = BigramNN(file_path,split_ratio)
