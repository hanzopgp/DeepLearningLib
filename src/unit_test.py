from global_imports import *

if __name__ == '__main__':
	gen2C = DataGen2Classes()
	gen2C.make_checker_board()
	gen2C.display_data()
	gen2C.make_2_gaussians(sigma=0.5)
	gen2C.display_data()
	gen2C.make_4_gaussians(sigma=0.5)
	gen2C.display_data()

	# gen4D = DataGenMultiClass(4)
	# gen4D.make_vertical()
	# gen4D.display_data()
	# gen4D.make_spiral()
	# gen4D.display_data()
	# gen4D.make_sinus()
	# gen4D.display_data()
