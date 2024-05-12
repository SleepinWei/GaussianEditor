import torch 
import open_clip
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, encoder_hidden_dims, decoder_hidden_dims):
        super(Autoencoder, self).__init__()
        encoder_layers = []
        for i in range(len(encoder_hidden_dims)):
            if i == 0:
                encoder_layers.append(nn.Linear(512, encoder_hidden_dims[i]))
            else:
                encoder_layers.append(torch.nn.BatchNorm1d(encoder_hidden_dims[i-1]))
                encoder_layers.append(nn.ReLU())
                encoder_layers.append(nn.Linear(encoder_hidden_dims[i-1], encoder_hidden_dims[i]))
        self.encoder = nn.ModuleList(encoder_layers)
             
        decoder_layers = []
        for i in range(len(decoder_hidden_dims)):
            if i == 0:
                decoder_layers.append(nn.Linear(encoder_hidden_dims[-1], decoder_hidden_dims[i]))
            else:
                encoder_layers.append(torch.nn.BatchNorm1d(decoder_hidden_dims[i-1]))
                decoder_layers.append(nn.ReLU())
                decoder_layers.append(nn.Linear(decoder_hidden_dims[i-1], decoder_hidden_dims[i]))
        self.decoder = nn.ModuleList(decoder_layers)
        # print(self.encoder, self.decoder)
    def forward(self, x):
        for m in self.encoder:
            x = m(x)
        x = x / x.norm(dim=-1, keepdim=True)
        for m in self.decoder:
            x = m(x)
        x = x / x.norm(dim=-1, keepdim=True)
        return x
    
    @torch.no_grad
    def encode(self, x):
        for m in self.encoder:
            x = m(x)    
        x = x / x.norm(dim=-1, keepdim=True)
        return x

    @torch.no_grad
    def decode(self, x):
        for m in self.decoder:
            x = m(x)    
        x = x / x.norm(dim=-1, keepdim=True)
        return x

class OpenCLIPEncoder:
    def __init__(self, vae_path:str=None):
        clip_model_type = "ViT-B-16"
        clip_model_pretrained = "laion2b_s34b_b88k"

        model, _, _ = open_clip.create_model_and_transforms(
            clip_model_type,
            pretrained=clip_model_pretrained,
            precision="fp16",
        )

        model.eval()
        self.tokenizer = open_clip.get_tokenizer(clip_model_type)
        self.model = model.to("cuda")
        # self.clip_n_dims = self.config.clip_n_dims

        # self.positives = self.positive_input.value.split(";")
        self.negatives = ("object", "things", "stuff", "texture")
        with torch.no_grad():
            # tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
            # self.pos_embeds = model.encode_text(tok_phrases)
            self.pos_embeds = None
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.negatives]).to("cuda")
            self.neg_embeds = model.encode_text(tok_phrases).to(torch.float32)

        # self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
        self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)

        if vae_path is None:
            self.vae = None
        else:
            encoder_hidden_dims = [256,128,64,32,3]
            decoder_hidden_dims = [16,32,64,128,256,256,512]
            self.vae = Autoencoder(encoder_hidden_dims,decoder_hidden_dims)
            self.vae.load_state_dict(torch.load(vae_path))
            self.vae = self.vae.to("cuda")
            self.vae.eval()

            # self.neg_embeds = self.vae.decode(self.neg_embeds)

    def get_relevancy(self, prompt:str, embed: torch.Tensor, positive_id: int) -> torch.Tensor:
        # embed: image embedding: H, W , languge_feat_len (3)
        embed_shape = embed.shape
        embed = embed.flatten(start_dim=0,end_dim=1)

        if self.vae is not None: 
            embed = self.vae.decode(embed)

        positives = prompt.split(";")
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in positives]).to("cuda")
            self.pos_embeds = self.model.encode_text(tok_phrases).to(torch.float32)
            self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)

                # self.pos_embeds = self.vae.decode(self.pos_embeds)

        phrases_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0)
        p = phrases_embeds.to(embed.dtype)  # phrases x 512
        output = torch.mm(embed, p.T)  # 1 x phrases
        positive_vals = output[..., positive_id : positive_id + 1]  # rays x 1
        negative_vals = output[..., len(positives) :]  # rays x N_phrase
        repeated_pos = positive_vals.repeat(1, len(self.negatives))  # rays x N_phrase

        sims = torch.stack((repeated_pos, negative_vals), dim=-1)  # rays x N-phrase x 2
        softmax = torch.softmax(10 * sims, dim=-1)  # rays x n-phrase x 2
        best_id = softmax[..., 0].argmin(dim=1)  # rays x 2 找到最高 score 对应哪个 negative val

        relevancy_map = torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(self.negatives), 2))[
            :, 0, :
        ]

        return relevancy_map.reshape((embed_shape[0],embed_shape[1],-1))