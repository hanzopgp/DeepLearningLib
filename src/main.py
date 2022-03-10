from data.data_generation import DataGeneration

if __name__ == '__main__':
	data_generation = DataGeneration(points=100, classes=2)
	data_generation.make_vertical_data()
	data_generation.display_data()

	data_generation = DataGeneration(points=100, classes=2)
	data_generation.make_spiral_data()
	data_generation.display_data()

	data_generation = DataGeneration(points=100)
	data_generation.make_sinus_data()
	data_generation.display_data()
