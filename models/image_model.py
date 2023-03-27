# -*- coding: utf-8 -*-

# Copyright (c) 2022-2022 cocone m, Inc.


class ImageModel:
    def __init__(self, model_path, model_id, config=None, **kwargs):
        self.model_path = model_path
        self.model_id = model_id
        self.config = config

        self.load_model()

    def load_model(self):
        raise NotImplementedError()

    def generate_images(self):
        raise NotImplementedError()
