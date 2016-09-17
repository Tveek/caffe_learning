import caffe
caffe.set_device(0)
caffe.set_mode_gpu()
solver = caffe.SGDSolver('./data/solver.prototxt')
solver.solve()

