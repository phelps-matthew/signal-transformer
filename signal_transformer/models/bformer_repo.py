class BendingCollegeWav2Vec(BaseProcess):
    """
    A more wav2vec 2.0 style of constrastive self-supervision, more inspired-by than exactly like it.
    """

    def __init__(
        self,
        encoder,
        context_fn,
        mask_rate=0.1,
        mask_span=6,
        learning_rate=0.01,
        temp=0.5,
        permuted_encodings=False,
        permuted_contexts=False,
        enc_feat_l2=0.001,
        multi_gpu=False,
        l2_weight_decay=1e-4,
        unmasked_negative_frac=0.25,
        encoder_grad_frac=1.0,
        num_negatives=100,
        **kwargs
    ):
        self.predict_length = mask_span
        self._enc_downsample = encoder.downsampling_factor
        if multi_gpu:
            encoder = nn.DataParallel(encoder)
            context_fn = nn.DataParallel(context_fn)
        if encoder_grad_frac < 1:
            encoder.register_backward_hook(
                lambda module, in_grad, out_grad: tuple(
                    encoder_grad_frac * ig for ig in in_grad
                )
            )
        super(BendingCollegeWav2Vec, self).__init__(
            encoder=encoder,
            context_fn=context_fn,
            loss_fn=nn.CrossEntropyLoss(),
            lr=learning_rate,
            l2_weight_decay=l2_weight_decay,
            metrics=dict(Accuracy=self._contrastive_accuracy, Mask_pct=self._mask_pct),
            **kwargs
        )
        self.best_metric = None
        self.mask_rate = mask_rate
        self.mask_span = mask_span
        self.temp = temp
        self.permuted_encodings = permuted_encodings
        self.permuted_contexts = permuted_contexts
        self.beta = enc_feat_l2
        self.start_token = getattr(context_fn, "start_token", None)
        self.unmasked_negative_frac = unmasked_negative_frac
        self.num_negatives = num_negatives

    def description(self, sequence_len):
        encoded_samples = self._enc_downsample(sequence_len)
        desc = "{} samples | mask span of {} at a rate of {} => E[masked] ~= {}".format(
            encoded_samples,
            self.mask_span,
            self.mask_rate,
            int(encoded_samples * self.mask_rate * self.mask_span),
        )
        return desc

    def _generate_negatives(self, z):
        """Generate negative samples to compare each sequence location against"""
        batch_size, feat, full_len = z.shape
        z_k = z.permute([0, 2, 1]).reshape(-1, feat)
        with torch.no_grad():
            # candidates = torch.arange(full_len).unsqueeze(-1).expand(-1, self.num_negatives).flatten()
            negative_inds = torch.randint(
                0, full_len - 1, size=(batch_size, full_len * self.num_negatives)
            )
            # From wav2vec 2.0 implementation, I don't understand
            # negative_inds[negative_inds >= candidates] += 1

            for i in range(1, batch_size):
                negative_inds[i] += i * full_len

        z_k = z_k[negative_inds.view(-1)].view(
            batch_size, full_len, self.num_negatives, feat
        )
        return z_k, negative_inds

    def _calculate_similarity(self, z, c, negatives):
        c = c[..., 1:].permute([0, 2, 1]).unsqueeze(-2)
        z = z.permute([0, 2, 1]).unsqueeze(-2)

        # In case the contextualizer matches exactly, need to avoid divide by zero errors
        negative_in_target = (c == negatives).all(-1)
        targets = torch.cat([c, negatives], dim=-2)

        logits = F.cosine_similarity(z, targets, dim=-1) / self.temp
        if negative_in_target.any():
            logits[1:][negative_in_target] = float("-inf")

        return logits.view(-1, logits.shape[-1])

    def forward(self, *inputs):
        z = self.encoder(inputs[0])

        if self.permuted_encodings:
            z = z.permute([1, 2, 0])

        unmasked_z = z.clone()

        batch_size, feat, samples = z.shape

        if self._training:
            mask = _make_mask(
                (batch_size, samples), self.mask_rate, samples, self.mask_span
            )
        else:
            mask = torch.zeros(
                (batch_size, samples), requires_grad=False, dtype=torch.bool
            )
            half_avg_num_seeds = max(1, int(samples * self.mask_rate * 0.5))
            if samples <= self.mask_span * half_avg_num_seeds:
                raise ValueError("Masking the entire span, pointless.")
            mask[
                :,
                _make_span_from_seeds(
                    (samples // half_avg_num_seeds)
                    * np.arange(half_avg_num_seeds).astype(int),
                    self.mask_span,
                ),
            ] = True

        c = self.context_fn(z, mask)

        # Select negative candidates and generate labels for which are correct labels
        negatives, negative_inds = self._generate_negatives(z)

        # Prediction -> batch_size x predict_length x predict_length
        logits = self._calculate_similarity(unmasked_z, c, negatives)
        return logits, z, mask

    @staticmethod
    def _mask_pct(inputs, outputs):
        return outputs[2].float().mean().item()

    @staticmethod
    def _contrastive_accuracy(inputs, outputs):
        logits = outputs[0]
        labels = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.long)
        return StandardClassification._simple_accuracy([labels], logits)

    def calculate_loss(self, inputs, outputs):
        logits = outputs[0]
        labels = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.long)
        # Note the loss_fn here integrates the softmax as per the normal classification pipeline (leveraging logsumexp)
        return self.loss_fn(logits, labels) + self.beta * outputs[1].pow(2).mean()
