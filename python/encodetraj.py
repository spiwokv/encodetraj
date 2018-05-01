def encodetrajectory(infilename='', intopname='', plotfilename='',
                     boxx=0.0, boxy=0.0, boxz=0.0, atestset=0.2,
                     shuffle=1, layers=2, layer1=256, layer2=256,
                     encdim=3, actfun1='sigmoid', actfun2='sigmoid'):
  # Loading trajectory
  try:
    traj = md.load(infilename, top=intopname)
  except:
    print "Cannot load %s or %s, exiting." % (infilename, intopname)
    parser.print_help()
    exit(0)
  else:
    print "%s succesfully loaded" % traj
  print
  
  # Ploting model scheme
  plotfiletype = 0
  if plotfilename != '':
    plotfiletype = 1
    if plotfilename[-4:] != '.png':
      plotfilename = plotfilename + '.png'
  
  # Conversion of the trajectory from Nframes x Natoms x 3 to Nframes x (Natoms x 3)
  trajsize = traj.xyz.shape
  traj2 = np.zeros((trajsize[0], trajsize[1]*3))
  for i in range(trajsize[1]):
    traj2[:,3*i]   = traj.xyz[:,i,0]
    traj2[:,3*i+1] = traj.xyz[:,i,1]
    traj2[:,3*i+2] = traj.xyz[:,i,2]
  
  # Checking whether all atoms fit the box
  if (np.amin(traj2)) < 0.0:
    print "ERROR: Some of atom has negative coordinate (i.e. it is outside the box)"
    exit(0)

  if boxx == 0.0 or boxy == 0.0 or boxz == 0.0:
    print "WARNING: box size not set, it will be determined automatically"
    if boxx == 0.0:
      boxx = 1.2*np.amax(traj.xyz[:,:,0])
    if boxy == 0.0:
      boxy = 1.2*np.amax(traj.xyz[:,:,1])
    if boxz == 0.0:
      boxz = 1.2*np.amax(traj.xyz[:,:,2])
    print "box size set to %6.3f x %6.3f x %6.3f nm" % (boxx, boxy, boxz)
    print
  
  if np.amax(traj.xyz[:,:,0]) > boxx or np.amax(traj.xyz[:,:,1]) > boxy or np.amax(traj.xyz[:,:,2]) > boxz:
    print "ERROR: Some of atom has coordinate higher than box size (i.e. it is outside the box)"
    exit(0)
  
  if boxx > 2.0*np.amax(traj.xyz[:,:,0]) or boxy > 2.0*np.amax(traj.xyz[:,:,0]) or boxz > 2.0*np.amax(traj.xyz[:,:,0]):
    print "WARNING: Box size is bigger than 2x of highest coordinate,"
    print "maybe the box is too big or the molecule is not centered"
  
  maxbox = max([boxx, boxy, boxz])

  # Splitting the trajectory into training and testing sets
  testsize = int(atestset * trajsize[0])
  if testsize < 1:
    print "ERROR: testset empty, increase testsize"
    exit(0)
  print "Training and test sets consist of %i and %i trajectory frames, respectively" % (trajsize[0]-testsize, testsize)
  print
  
  # Shuffling the trajectory before splitting
  if shuffle == 1:
    print "Trajectory will be shuffled before splitting into training and test set"
  elif shuffle == 0:
    print "Trajectory will NOT be shuffled before splitting into training and test set"
    print "(first %i frames will be used for trainintg, next %i for testing)" % (trajsize[0]-testsize, testsize)
  indexes = range(trajsize[0])
  if shuffle == 1:
    np.random.shuffle(indexes)
  training_set, testing_set = traj2[indexes[:-testsize],:]/maxbox, traj2[indexes[-testsize:],:]/maxbox
  
  # (Deep) learning  
  input_coord = krs.layers.Input(shape=(trajsize[1]*3,))
  encoded = krs.layers.Dense(args.layer1, activation=args.actfun1, use_bias=True)(input_coord)
  if args.layers == 3:
    encoded = krs.layers.Dense(args.layer2, activation=args.actfun2, use_bias=True)(encoded)
  encoded = krs.layers.Dense(args.encdim, activation='linear', use_bias=True)(encoded)
  if args.layers == 3:
    encoded = krs.layers.Dense(args.layer2, activation=args.actfun2, use_bias=True)(encoded)
  decoded = krs.layers.Dense(args.layer1, activation=args.actfun1, use_bias=True)(encoded)
  decoded = krs.layers.Dense(trajsize[1]*3, activation='linear', use_bias=True)(decoded)
  autoencoder = krs.models.Model(input_coord, decoded)
  
  encoder = krs.models.Model(input_coord, encoded)
  
  encoded_input = krs.layers.Input(shape=(args.encdim,))
  if args.layers == 2:
    decoder_layer1 = autoencoder.layers[-2]
    decoder_layer2 = autoencoder.layers[-1]
    decoder = krs.models.Model(encoded_input, decoder_layer2(decoder_layer1(encoded_input)))
  if args.layers == 3:
    decoder_layer1 = autoencoder.layers[-3]
    decoder_layer2 = autoencoder.layers[-2]
    decoder_layer3 = autoencoder.layers[-1]
    decoder = krs.models.Model(encoded_input, decoder_layer3(decoder_layer2(decoder_layer1(encoded_input))))
  
  autoencoder.compile(optimizer=args.optim, loss=args.loss)
  
  if args.batch>0:
    autoencoder.fit(training_set, training_set,
                    epochs=args.epochs,
                    batch_size=args.batch,
                    validation_data=(testing_set, testing_set))
  else:
    autoencoder.fit(training_set, training_set,
                    epochs=args.epochs,
                    validation_data=(testing_set, testing_set))
  
  # Encoding and decoding the trajectory
  encoded_coords = encoder.predict(traj2/maxbox)
  decoded_coords = decoder.predict(encoded_coords)
  
  # Calculating Pearson correlation coefficient
  vec1 = traj2.reshape((trajsize[0]*trajsize[1]*3))
  vec2 = decoded_coords.reshape((trajsize[0]*trajsize[1]*3))*maxbox
  print
  print "Pearson correlation coefficient for encoded-decoded trajectory is %f" % np.corrcoef(vec1,vec2)[0,1]
  print
  
  #training_set, testing_set = traj2[indexes[:-testsize],:]/maxbox, traj2[indexes[-testsize:],:]/maxbox
  vec1 = traj2[indexes[:-testsize],:].reshape(((trajsize[0]-testsize)*trajsize[1]*3))
  vec2 = decoded_coords[indexes[:-testsize],:].reshape(((trajsize[0]-testsize)*trajsize[1]*3))*maxbox
  print "Pearson correlation coefficient for encoded-decoded training set is %f" % np.corrcoef(vec1,vec2)[0,1]
  print
  
  vec1 = traj2[indexes[-testsize:],:].reshape((testsize*trajsize[1]*3))
  vec2 = decoded_coords[indexes[-testsize:],:].reshape((testsize*trajsize[1]*3))*maxbox
  print "Pearson correlation coefficient for encoded-decoded testing set is %f" % np.corrcoef(vec1,vec2)[0,1]
  print
  
  # Generating output
  lowfiletype = 0
  highfiletype = 0
  if args.lowfile != '':
    if args.lowfile[-4:] == '.xvg':
      lowfilename = args.lowfile
      lowfiletype = 1
    elif args.lowfile[-4:] == '.txt':
      lowfilename = args.lowfile
      lowfiletype = 2
    else:
      lowfilename = args.lowfile + '.txt'
      lowfiletype = 2
  if args.highfile != '':
    if args.highfile[-4:] == '.xvg':
      highfilename = args.highfile
      highfiletype = 1
    elif args.highfile[-4:] == '.txt':
      highfilename = args.highfile
      highfiletype = 2
    else:
      highfilename = args.highfile + '.txt'
      highfiletype = 2
  
  # Generating low-dimensional output
  if lowfiletype > 0:
    print "Writing low-dimensional embeddings into %s" % lowfilename
    print
    if lowfiletype == 1:
      ofile = open(lowfilename, "w")
      ofile.write("# This file was created on %s\n" % dt.datetime.now().isoformat())
      ofile.write("# Created by: encodetraj.py V 0.1\n")
      sysargv = ""
      for item in sys.argv:
        sysargv = sysargv+item+" "
      ofile.write("# Command line: %s\n" % sysargv)
      ofile.write("@TYPE xy\n")
      ofile.write("@ title \"Autoencoded trajectory\"\n")
      ofile.write("@ xaxis  label \"low-dimensional embedding 1\"\n")
      ofile.write("@ yaxis  label \"low-dimensional embedding 2\"\n")
      for i in range(trajsize[0]):
        for j in range(args.encdim):
          ofile.write("%f " % encoded_coords[i][j])
        typeofset = 'TE'
        if i in indexes[:-testsize]:
          typeofset = 'TR'
        ofile.write("%s \n" % typeofset)
      ofile.close()
    if lowfiletype == 2:
      ofile = open(lowfilename, "w")
      for i in range(trajsize[0]):
        for j in range(args.encdim):
          ofile.write("%f " % encoded_coords[i][j])
        typeofset = 'TE'
        if i in indexes[:-testsize]:
          typeofset = 'TR'
        ofile.write("%s \n" % typeofset)
      ofile.close()
  
  # Generating high-dimensional output
  if highfiletype > 0:
    print "Writing original and encoded-decoded coordinates into %s" % highfilename
    print
    if highfiletype == 1:
      ofile = open(highfilename, "w")
      ofile.write("# This file was created on %s\n" % dt.datetime.now().isoformat())
      ofile.write("# Created by: encodetraj.py V 0.1\n")
      sysargv = ''
      for item in sys.argv:
        sysargv = sysargv+item+' '
      ofile.write("# Command line: %s\n" % sysargv)
      ofile.write("@TYPE xy\n")
      ofile.write("@ title \"Autoencoded and decoded trajectory\"\n")
      ofile.write("@ xaxis  label \"original coordinate\"\n")
      ofile.write("@ yaxis  label \"encoded and decoded coordinate\"\n")
      for i in range(trajsize[0]):
        for j in range(trajsize[1]*3):
          ofile.write("%f %f " % (traj2[i][j], decoded_coords[i][j]*maxbox))
          typeofset = 'TE'
          if i in indexes[:-testsize]:
            typeofset = 'TR'
          ofile.write("%s \n" % typeofset)
      ofile.close()
    if highfiletype == 2:
      ofile = open(highfilename, "w")
      for i in range(trajsize[0]):
        for j in range(trajsize[1]*3):
          ofile.write("%f %f " % (traj2[i][j], decoded_coords[i][j]*maxbox))
          typeofset = 'TE'
          if i in indexes[:-testsize]:
            typeofset = 'TR'
          ofile.write("%s \n" % typeofset)
      ofile.close()
  
  # Generating filtered trajectory
  if args.filterfile != '':
    filterfilename = args.filterfile
    if filterfilename[-4:] != '.xtc':
      filterfilename = filterfilename + '.xtc'
    print "Writing encoded-decoded trajectory into %s" % highfilename
    print
    decoded_coords2 = np.zeros((trajsize[0], trajsize[1], 3))
    for i in range(trajsize[1]):
      decoded_coords2[:,i,0] = decoded_coords[:,3*i]*maxbox
      decoded_coords2[:,i,1] = decoded_coords[:,3*i+1]*maxbox
      decoded_coords2[:,i,2] = decoded_coords[:,3*i+2]*maxbox
    traj.xyz = decoded_coords2
    traj.save_xtc(filterfilename)
  
  # Saving a plot of the model
  if plotfiletype == 1:
    krs.utils.plot_model(autoencoder, to_file=plotfilename)
  
  # Saving the model
  if args.modelfile != '':
    print "Writing model into %s.txt" % args.modelfile
    print
    ofile = open(args.modelfile+'.txt', "w")
    ofile.write("maxbox = %f\n" % maxbox)
    ofile.write("input_coord = krs.layers.Input(shape=(trajsize[1]*3,))\n")
    ofile.write("encoded = krs.layers.Dense(%i, activation='%s', use_bias=True)(input_coord)\n" % (args.layer1, args.actfun1))
    if args.layers == 3:
      ofile.write("encoded = krs.layers.Dense(%i, activation='%s', use_bias=True)(encoded)\n" % (args.layer2, args.actfun2))
    ofile.write("encoded = krs.layers.Dense(%i, activation='linear', use_bias=True)(encoded)\n" % args.encdim)
    if args.layers == 3:
      ofile.write("encoded = krs.layers.Dense(%i, activation='%s', use_bias=True)(encoded)\n" % (args.layer2, args.actfun2))
    ofile.write("decoded = krs.layers.Dense(%i, activation='%s', use_bias=True)(encoded)\n" % (args.layer1, args.actfun1))
    ofile.write("decoded = krs.layers.Dense(trajsize[1]*3, activation='linear', use_bias=True)(decoded)\n")
    ofile.write("autoencoder = krs.models.Model(input_coord, decoded)\n")
    ofile.close()
    print "Writing model weights and biases into %s_*.npy NumPy arrays" % args.modelfile
    print
    if args.layers == 2:
      np.save(file=args.modelfile+"_1.npy", arr=autoencoder.layers[1].get_weights())
      np.save(file=args.modelfile+"_2.npy", arr=autoencoder.layers[2].get_weights())
      np.save(file=args.modelfile+"_3.npy", arr=autoencoder.layers[3].get_weights())
      np.save(file=args.modelfile+"_4.npy", arr=autoencoder.layers[4].get_weights())
    else:
      np.save(file=args.modelfile+"_1.npy", arr=autoencoder.layers[1].get_weights())
      np.save(file=args.modelfile+"_2.npy", arr=autoencoder.layers[2].get_weights())
      np.save(file=args.modelfile+"_3.npy", arr=autoencoder.layers[3].get_weights())
      np.save(file=args.modelfile+"_4.npy", arr=autoencoder.layers[4].get_weights())
      np.save(file=args.modelfile+"_5.npy", arr=autoencoder.layers[5].get_weights())
      np.save(file=args.modelfile+"_6.npy", arr=autoencoder.layers[6].get_weights())
  
  # Saving collective motions trajectories
  #if args.collectivefile != '':
  #  collectivefile = args.collectivefile
  #  if collectivefile[-4:] == '.xtc':
  #    collectivefile = collectivefile[:-4]
  #  traj = traj[:args.ncollective]
  #  print "Writing collective motion into %s_1.xtc" % collectivefile
  #  print
  #  collective = np.zeros((args.ncollective, 3))
  #  cvmin = np.amin(encoded_coords[:,0])
  #  cvmax = np.amax(encoded_coords[:,0])
  #  for i in range(args.ncollective):
  #    collective[i,0] = cvmin+(cvmax-cvmin)*float(i)/float(args.ncollective-1)
  #    collective[i,1] = np.mean(encoded_coords[:,1])
  #    collective[i,2] = np.mean(encoded_coords[:,2])
  #  collective2 = decoder.predict(collective)
  #  collective3 = np.zeros((args.ncollective, trajsize[1], 3))
  #  for i in range(trajsize[1]):
  #    collective3[:,i,0] = collective2[:,3*i]*maxbox
  #    collective3[:,i,1] = collective2[:,3*i+1]*maxbox
  #    collective3[:,i,2] = collective2[:,3*i+2]*maxbox
  #  traj.xyz = collective3
  #  traj.save_xtc(collectivefile+"_1.xtc")
  #  print "Writing collective motion into %s_2.xtc" % collectivefile
  #  print
  #  collective = np.zeros((args.ncollective, 3))
  #  cvmin = np.amin(encoded_coords[:,1])
  #  cvmax = np.amax(encoded_coords[:,1])
  #  for i in range(args.ncollective):
  #    collective[i,0] = np.mean(encoded_coords[:,0])
  #    collective[i,1] = cvmin+(cvmax-cvmin)*float(i)/float(args.ncollective-1)
  #    collective[i,2] = np.mean(encoded_coords[:,2])
  #  collective2 = decoder.predict(collective)
  #  collective3 = np.zeros((args.ncollective, trajsize[1], 3))
  #  for i in range(trajsize[1]):
  #    collective3[:,i,0] = collective2[:,3*i]*maxbox
  #    collective3[:,i,1] = collective2[:,3*i+1]*maxbox
  #    collective3[:,i,2] = collective2[:,3*i+2]*maxbox
  #  traj.xyz = collective3
  #  traj.save_xtc(collectivefile+"_2.xtc")
  #  print "Writing collective motion into %s_3.xtc" % collectivefile
  #  print
  #  collective = np.zeros((args.ncollective, 3))
  #  cvmin = np.amin(encoded_coords[:,2])
  #  cvmax = np.amax(encoded_coords[:,2])
  #  for i in range(args.ncollective):
  #    collective[i,0] = np.mean(encoded_coords[:,0])
  #    collective[i,1] = np.mean(encoded_coords[:,1])
  #    collective[i,2] = cvmin+(cvmax-cvmin)*float(i)/float(args.ncollective-1)
  #  collective2 = decoder.predict(collective)
  #  collective3 = np.zeros((args.ncollective, trajsize[1], 3))
  #  for i in range(trajsize[1]):
  #    collective3[:,i,0] = collective2[:,3*i]*maxbox
  #    collective3[:,i,1] = collective2[:,3*i+1]*maxbox
  #    collective3[:,i,2] = collective2[:,3*i+2]*maxbox
  #  traj.xyz = collective3
  #  traj.save_xtc(collectivefile+"_3.xtc")
  #
  
if __name__ == "__main__":
  # Loading necessary libraries
  libnames = [('mdtraj', 'md'), ('numpy', 'np'), ('keras', 'krs'), ('argparse', 'arg'), ('datetime', 'dt'), ('sys', 'sys')]
  
  for (name, short) in libnames:
    try:
      lib = __import__(name)
    except:
      print "Library %s is not installed, exiting" % name
      exit(0)
    else:
      globals()[short] = lib
  
  # Parsing command line arguments
  parser = arg.ArgumentParser(description='(Deep learning) autoencoders for molecular trajectory analysis, requires numpy, keras and mdtraj')
  
  parser.add_argument('-i', dest='infile', default='traj.xtc',
  help='Input trajectory in pdb, xtc, trr, dcd, netcdf or mdcrd, WARNING: the trajectory must be 1. centered in the PBC box, 2. fitted to a reference structure and 3. must contain only atoms to be analysed!')
  
  parser.add_argument('-p', dest='intop', default='top.pdb',
  help='Input topology in pdb, WARNING: the structure must be 1. centered in the PBC box and 2. must contain only atoms to be analysed!')
  
  parser.add_argument('-boxx', dest='boxx', default=0.0, type=float,
  help='Size of x coordinate of PBC box (from 0 to set value in nm)')
  
  parser.add_argument('-boxy', dest='boxy', default=0.0, type=float,
  help='Size of y coordinate of PBC box (from 0 to set value in nm)')
  
  parser.add_argument('-boxz', dest='boxz', default=0.0, type=float,
  help='Size of z coordinate of PBC box (from 0 to set value in nm)')
  
  parser.add_argument('-testset', dest='testset', default=0.10, type=float,
  help='Size of test set (fraction of the trajectory, default = 0.1)')
  
  parser.add_argument('-shuffle', dest='shuffle', default='True',
  help='Shuffle trajectory frames to obtain training and test set (default True)')
  
  parser.add_argument('-layers', dest='layers', default=2, type=int,
  help='Number of encoding layers (same as number of decoding, allowed values 2-3, default = 2)')
  
  parser.add_argument('-layer1', dest='layer1', default=256, type=int,
  help='Number of neurons in the second encoding layer (default = 256)')
  
  parser.add_argument('-layer2', dest='layer2', default=256, type=int,
  help='Number of neurons in the third encoding layer (default = 256)')
  
  parser.add_argument('-encdim', dest='encdim', default=3, type=int,
  help='Encoding dimension (default = 3)')
  
  parser.add_argument('-actfun1', dest='actfun1', default='sigmoid',
  help='Activation function of the first layer (default = sigmoid, for options see keras documentation)')
  
  parser.add_argument('-actfun2', dest='actfun2', default='sigmoid',
  help='Activation function of the second layer (default = sigmoid, for options see keras documentation)')
  
  parser.add_argument('-optim', dest='optim', default='adam',
  help='Optimizer (default = adam, for options see keras documentation)')
  
  parser.add_argument('-loss', dest='loss', default='mean_squared_error',
  help='Loss function (default = mean_squared_error, for options see keras documentation)')
  
  parser.add_argument('-epochs', dest='epochs', default=100, type=int,
  help='Number of epochs (default = 100, >1000 may be necessary for real life applications)')
  
  parser.add_argument('-batch', dest='batch', default=256, type=int,
  help='Batch size (0 = no batches, default = 256)')
  
  parser.add_argument('-low', dest='lowfile', default='',
  help='Output file with low-dimensional embedings (xvg or txt, default = no output)')
  
  parser.add_argument('-high', dest='highfile', default='',
  help='Output file with original coordinates and encoded-decoded coordinates (xvg or txt, default = no output)')
  
  parser.add_argument('-filter', dest='filterfile', default='',
  help='Output file with encoded-decoded trajectory in .xtc format (default = no output)')
  
  # Extraction of collective motions does not work very well
  #parser.add_argument('-collective', dest='collectivefile', default='',
  #help='Output files with collective motions trajectories in .xtc format (default = no output)')
  #
  #parser.add_argument('-ncollective', dest='ncollective', default=10, type=int,
  #help='Number of frames in collective motions trajectories (default = 20)')
  
  parser.add_argument('-model', dest='modelfile', default='',
  help='Prefix for output model files (experimental, default = no output)')
  
  parser.add_argument('-plot', dest='plotfile', default='',
  help='Model plot file in png or svg (default = no output)')
  
  parser.add_argument('-plumed', dest='plumedfile', default='',
  help='Output file for Plumed (default = no output)')
  
  args = parser.parse_args()
  boxx = args.boxx
  boxy = args.boxy
  boxz = args.boxz

  if args.testset < 0.0 or args.testset > 0.5:
    print "ERROR: -testset must be 0.0 - 0.5"
    exit(0)
  atestset = int(args.testset)
  
  # Shuffling the trajectory before splitting
  if args.shuffle == "True":
    shuffle = 1
  elif args.shuffle == "False":
    shuffle = 0
  else:
    print "ERROR: -shuffle %s not understood" % args.shuffle
    exit(0)
  if args.layers < 2 or args.layers > 3:
    print "ERROR: -layers must be 2-3, for deeper learning contact authors"
    exit(0)
  if args.layer1 > 1024:
    print "WARNING: You plan to use %i neurons in the second layer, could be slow"
  if args.layers == 3:
    if args.layer2 > 1024:
      print "WARNING: You plan to use %i neurons in the third layer, could be slow"
  if args.actfun1 not in ['softmax','elu','selu','softplus','softsign','relu','tanh','sigmoid','hard_sigmoid','linear']:
    print "ERROR: cannot understand -actfun1 %s" % args.actfun1
    exit(0)
  if args.layers == 3:
    if args.actfun2 not in ['softmax','elu','selu','softplus','softsign','relu','tanh','sigmoid','hard_sigmoid','linear']:
      print "ERROR: cannot understand -actfun2 %s" % args.actfun1
      exit(0)
  layers = args.layers
  layer1 = args.layer1
  layer2 = args.layer2
  encdim = args.encdim
  actfun1 = args.actfun1 
  actfun2 = args.actfun2

