import unittest

import ood

class TestUtilsMethods(unittest.TestCase):
    def test_dataset_dims(self):
        ds = ood.utils.load_data("mnist", 64)
        self.assertEqual(ds['x_dim'], 784)
        self.assertEqual(ds['y_dim'], 10)
    
    def test_dataset_batch_size(self):
        bs = 64
        ds = ood.utils.load_data("mnist", bs)
        _, y = next(iter(ds['dataloader']))
        self.assertEqual(len(y), bs)

    def test_class_loading(self):
        ds = ood.utils.load_data("mnist", 64, classes=[1])
        self.assertTrue((ds['dataloader'].dataset.targets == 1).all().item())
        ds = ood.utils.load_data("mnist", 64, classes=[1, 2])
        self.assertAlmostEqual((ds['dataloader'].dataset.targets == 1).sum().item() / len(ds['dataloader'].dataset.targets), 0.5, 1)
        self.assertAlmostEqual((ds['dataloader'].dataset.targets == 2).sum().item() / len(ds['dataloader'].dataset.targets), 0.5, 1)
    
    def test_train_test_loading(self):
        ds = ood.utils.load_data("mnist", 64, train=True)
        self.assertEqual(len(ds['dataloader'].dataset.targets), 60_000)
        ds = ood.utils.load_data("mnist", 64, train=False)
        self.assertEqual(len(ds['dataloader'].dataset.targets), 10_000)

    def test_model(self):
        params = {"num_in": 2, "num_out": 2, "params_mul": 10}
        model = ood.utils.load_model("softmax", 0.01, "cpu", params)
        self.assertEqual(model.features[0].in_features, params['num_in'])
        self.assertEqual(model.features[0].out_features, params['num_in'] * params["params_mul"])
        self.assertEqual(model.features[2].in_features, params['num_in'] * params["params_mul"])
        self.assertEqual(model.features[2].out_features, params['num_out'])

    def test_benchmark(self):
        ds = ood.utils.load_data("mnist", 64, train=False)
        model = ood.utils.load_model("softmax", 0.01, "cpu", {"num_in": ds["x_dim"], "params_mul": 10, "num_out": ds["y_dim"]})
        bm1 = ood.utils.benchmark(model, ds['dataloader'])
        model.fit(ood.utils.load_data("mnist", 64, train=True)['dataloader'], epochs=10)
        bm2 = ood.utils.benchmark(model, ds['dataloader'])
        self.assertGreater(bm2['accuracy'], bm1['accuracy'])
        self.assertLess(bm2['loss'], bm1['loss'])

if __name__ == '__main__':
    unittest.main()