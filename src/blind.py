import torch, torchvision
from torch.utils.data import DataLoader
import random
from unlearn import blindspot_unlearner

from main import *

transform_train = torchvision.transforms.Compose([
 	torchvision.transforms.Resize(224),
	torchvision.transforms.ToTensor(),
 	torchvision.transforms.Normalize((0.1307), (0.3081)),
 ])

'''transform_train = torchvision.transforms.Compose([
	torchvision.transforms.RandomCrop(224),
	torchvision.transforms.ToTensor(), # first, convert image to PyTorch tensor
	torchvision.transforms.Normalize((0.1307),(0.3081)) # normalize inputs
]),'''


if __name__ == '__main__':
	# Parse arguments
	opt = parse_args()
	seed_everything(seed=opt.seed)

	# Setup logger
	console_logger = get_logger(folder=opt.log_dir+'/'+opt.exp_name+'/')
	
	# Handle fixed class orders. Note: Class ordering code hacky. Would need to manually adjust here to test for different datasets.
	console_logger.debug("==> Loading dataset..")
	class_order = [87, 0, 52, 58, 44, 91, 68, 97, 51, 15, 94, 92, 10, 72, 49, 78, 61, 14, 8, 86, 84, 96, 18, 24, 32, 45, 88, 11, 4, 67, 69, 66, 77, 47, 79, 93, 29, 50, 57, 83, 17, 81, 41, 12, 37, 59, 25, 20, 80, 73, 1, 28, 6, 46, 62, 82, 53, 9, 31, 75, 38, 63, 33, 74, 27, 22, 36, 3, 16, 21, 60, 19, 70, 90, 89, 43, 5, 42, 65, 76, 40, 30, 23, 85, 2, 95, 56, 48, 71, 64, 98, 13, 99, 7, 34, 55, 54, 26, 35, 39] #Currently testing using iCARL test order-- restricted to CIFAR100. For the other class orders refer to https://github.com/arthurdouillard/incremental_learning.pytorch/tree/master/options/data
	if opt.dataset != 'CIFAR100' and opt.dataset !='ImageNet100': class_order=None

	# Handle 'path does not exist errors' 
	if not os.path.isdir(opt.log_dir+'/'+opt.exp_name):
		os.mkdir(opt.log_dir+'/'+opt.exp_name)
	if opt.old_exp_name!='test' and not os.path.isdir(opt.log_dir+'/'+opt.old_exp_name):
		os.mkdir(opt.log_dir+'/'+opt.old_exp_name)    

	# Set pretraining and continual dataloaders
	dobj = VisionDataset(opt, class_order=class_order)
	dobj.gen_cl_mapping()

	# Scenario #1: First pretraining on n classes, then do continual learning
	if opt.num_pretrain_classes > 0: 
		opt.num_classes = opt.num_pretrain_classes
		if opt.inp_size == 28: model = getattr(mnist, opt.model)(opt)
		if opt.inp_size == 32 or opt.inp_size == 64: model = getattr(cifar, opt.model)(opt)
		if opt.inp_size ==224: model = getattr(imagenet, opt.model)(opt)      
		if not os.path.isfile(opt.log_dir+opt.old_exp_name+'/pretrained_model.pth.tar'):
			console_logger.debug("==> Starting pre-training..") 
			# _, model = experiment(opt=opt, class_mask=torch.ones(opt.num_classes,opt.num_classes).cuda(), train_loader=dobj.pretrain_loader, \
			_, model = experiment(opt=opt, class_mask=torch.ones(opt.num_classes,opt.num_classes), train_loader=dobj.pretrain_loader, \
									test_loader=dobj.pretest_loader, model=model, logger=console_logger, num_passes=opt.num_pretrain_passes)
			save_model(opt, model) # Saves the pretrained model. Subsequent CL experiments directly load the pretrained model.
		else:
			model = load_model(opt, model, console_logger)
		# Reset the final block to new set of classes
		opt.num_classes = opt.num_classes_per_task*opt.num_tasks
		model.final = FinalBlock(opt, model.dim_out)
	
	# Scenario #2: Directly do continual learning from scratch
	else: 
		opt.num_classes = opt.num_classes_per_task*opt.num_tasks
		if opt.inp_size == 28: model = getattr(mnist, opt.model)(opt)
		if opt.inp_size == 32 or opt.inp_size == 64: model = getattr(cifar, opt.model)(opt)
		if opt.inp_size ==224: model = getattr(imagenet, opt.model)(opt)
		
	train_ds = torchvision.datasets.MNIST(root='.', train=True,download=True, transform=transform_train)
	valid_ds = torchvision.datasets.MNIST(root='.', train=False,download=True, transform=transform_train)

	batch_size = 16

	train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=0, pin_memory=True)
	valid_dl = DataLoader(valid_ds, batch_size, num_workers=0, pin_memory=True)

	num_classes = 10
	classwise_train = {}
	for i in range(num_classes):
		classwise_train[i] = []

	# print(train_ds.__getitem__(0))

	for img, label in train_ds:
		classwise_train[label].append((img, label))

	classwise_test = {}
	for i in range(num_classes):
		classwise_test[i] = []

	for img, label in valid_ds:
		classwise_test[label].append((img, label))

	# Getting the forget and retain validation data
	forget_valid = []
	forget_classes = [9]
	for cls in range(num_classes):
		if cls in forget_classes:
			for img, label in classwise_test[cls]:
				forget_valid.append((img, label))

	retain_valid = []
	for cls in range(num_classes):
		if cls not in forget_classes:
			for img, label in classwise_test[cls]:
				retain_valid.append((img, label))
				
	forget_train = []
	for cls in range(num_classes):
		if cls in forget_classes:
			for img, label in classwise_train[cls]:
				forget_train.append((img, label))

	retain_train = []
	for cls in range(num_classes):
		if cls not in forget_classes:
			for img, label in classwise_train[cls]:
				retain_train.append((img, label))

	forget_valid_dl = DataLoader(forget_valid, batch_size, num_workers=0, pin_memory=True)

	retain_valid_dl = DataLoader(retain_valid, batch_size, num_workers=0, pin_memory=True)

	forget_train_dl = DataLoader(forget_train, batch_size, num_workers=0, pin_memory=True)
	retain_train_dl = DataLoader(retain_train, batch_size, num_workers=0, pin_memory=True, shuffle = True)
	retain_train_subset = random.sample(retain_train, int(0.3*len(retain_train)))
	retain_train_subset_dl = DataLoader(retain_train_subset, batch_size, num_workers=0, pin_memory=True, shuffle = True)

	logger=console_logger

	console_logger.debug("==> Starting Learning Training..")

	best_prec1 = 0.0
	best_prec2 = 0.0
	best_prec3 = 0.0
	model = model.cuda() # Better speed with little loss in accuracy. If loss in accuracy is big, use apex.
	criterion = nn.CrossEntropyLoss().cuda()
	optimizer = optim.SGD(model.parameters(), lr=opt.maxlr, momentum=0.9, weight_decay=opt.weight_decay)
	scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2, eta_min=opt.minlr)
	class_mask = torch.zeros(opt.num_classes, opt.num_classes).cuda()
	
	num_passes=10

	# Train and test loop
	logger.info("==> Opts for this training: "+str(opt))

	for epoch in range(num_passes):
		# Handle lr scheduling
		if epoch <= 0: # Warm start of 1 epoch
			for param_group in optimizer.param_groups:
				param_group['lr'] = opt.maxlr * 0.1
		elif epoch == 1: # Then set to maxlr
			for param_group in optimizer.param_groups:
				param_group['lr'] = opt.maxlr
		else: # Aand go!
			scheduler.step()

		# Train and test loop
		logger.info("==> Starting pass number: "+str(epoch)+", Learning rate: " + str(optimizer.param_groups[0]['lr']))
		model, optimizer = train(opt=opt, loader=train_dl, model=model, criterion=criterion, optimizer=optimizer, epoch=epoch, logger=logger)
		prec1 = test(loader=valid_dl, model=model, criterion=criterion, class_mask=class_mask, logger=logger, epoch=epoch)
		prec2 = test(loader=retain_valid_dl, model=model, criterion=criterion, class_mask=class_mask, logger=logger, epoch=epoch)
		prec3 = test(loader=forget_valid_dl, model=model, criterion=criterion, class_mask=class_mask, logger=logger, epoch=epoch)

		# Log performance
		logger.info('==> Current total accuracy: [{:.3f}]\t'.format(prec1))
		logger.info('==> Current retian accuracy: [{:.3f}]\t'.format(prec2))
		logger.info('==> Current forget accuracy: [{:.3f}]\t'.format(prec3))
		if prec1 > best_prec1:
			logger.info('==> Total Accuracies\tPrevious: [{:.3f}]\t'.format(best_prec1) + 'Current: [{:.3f}]\t'.format(prec1))
			best_prec1 = float(prec1)
		if prec2 > best_prec2:
			logger.info('==> Retain Accuracies\tPrevious: [{:.3f}]\t'.format(best_prec2) + 'Current: [{:.3f}]\t'.format(prec2))
			best_prec3 = float(prec2)
		if prec3 > best_prec3:
			logger.info('==> Forget Accuracies\tPrevious: [{:.3f}]\t'.format(best_prec3) + 'Current: [{:.3f}]\t'.format(prec3))
			best_prec4 = float(prec3)

	logger.info('==> Training completed!\n\tTotal Acc: [{0:.3f}]\n\Retian Acc: [{1:.3f}]\n\Forget Acc: [{2:.3f}]'.format(best_prec1, best_prec2, best_prec3))
	
	# Log performance
	logger.info('==> Current total accuracy: [{:.3f}]\t'.format(prec1))
	logger.info('==> Current retian accuracy: [{:.3f}]\t'.format(prec2))
	logger.info('==> Current forget accuracy: [{:.3f}]\t'.format(prec3))
	if prec1 > best_prec1:
		logger.info('==> Total Accuracies\tPrevious: [{:.3f}]\t'.format(best_prec1) + 'Current: [{:.3f}]\t'.format(prec1))
		best_prec1 = float(prec1)
	if prec2 > best_prec2:
		logger.info('==> Retain Accuracies\tPrevious: [{:.3f}]\t'.format(best_prec2) + 'Current: [{:.3f}]\t'.format(prec2))
		best_prec3 = float(prec2)
	if prec3 > best_prec3:
		logger.info('==> Forget Accuracies\tPrevious: [{:.3f}]\t'.format(best_prec3) + 'Current: [{:.3f}]\t'.format(prec3))
		best_prec4 = float(prec3)

	console_logger.debug("==> Completed!")

	torch.save(model.state_dict(), "MNIST_epochs_10.pt")

	'''device = 'cuda'
	unlearning_teacher = model.cuda().eval()
	student_model = model.cuda()
	student_model.load_state_dict(torch.load("MNIST_epochs_10.pt", map_location = device))
	model = model.eval()'''
	device = 'cuda'
	unlearning_teacher = getattr(mnist, opt.model)(opt)
	unlearning_teacher = unlearning_teacher.cuda().eval()

	student_model = getattr(mnist, opt.model)(opt)
	student_model = student_model.cuda()
	student_model.load_state_dict(torch.load("MNIST_epochs_10.pt", map_location = device))
	model = model.eval()

	KL_temperature = 1

	optimizer = torch.optim.Adam(student_model.parameters(), lr = 0.0001)

	blindspot_unlearner(model = student_model, unlearning_teacher = unlearning_teacher, full_trained_teacher = model, 
          retain_data = retain_train_subset, forget_data = forget_train, epochs = 50, optimizer = optimizer, lr = 0.0001, 
          batch_size = 16, num_workers = 2, device = device, KL_temperature = KL_temperature)

	console_logger.debug("==> Completed!")
	console_logger.debug("==> Accuracy after 9 removed Completed!")

	console_logger.debug("==> model")
	prec1 = test(loader=valid_dl, model=model, criterion=criterion, class_mask=class_mask, logger=logger, epoch=epoch)
	prec2 = test(loader=retain_valid_dl, model=model, criterion=criterion, class_mask=class_mask, logger=logger, epoch=epoch)
	prec3 = test(loader=forget_valid_dl, model=model, criterion=criterion, class_mask=class_mask, logger=logger, epoch=epoch)

	logger.info('==> Total Accuracies\tPrevious: [{:.3f}]\t'.format(best_prec1) + 'Current: [{:.3f}]\t'.format(prec1))
	logger.info('==> Retain Accuracies\tPrevious: [{:.3f}]\t'.format(best_prec2) + 'Current: [{:.3f}]\t'.format(prec2))
	logger.info('==> Forget Accuracies\tPrevious: [{:.3f}]\t'.format(best_prec3) + 'Current: [{:.3f}]\t'.format(prec3))


	console_logger.debug("==> student_model")
	prec1 = test(loader=valid_dl, model=student_model, criterion=criterion, class_mask=class_mask, logger=logger, epoch=epoch)
	prec2 = test(loader=retain_valid_dl, model=student_model, criterion=criterion, class_mask=class_mask, logger=logger, epoch=epoch)
	prec3 = test(loader=forget_valid_dl, model=student_model, criterion=criterion, class_mask=class_mask, logger=logger, epoch=epoch)

	logger.info('==> Total Accuracies\tPrevious: [{:.3f}]\t'.format(best_prec1) + 'Current: [{:.3f}]\t'.format(prec1))
	logger.info('==> Retain Accuracies\tPrevious: [{:.3f}]\t'.format(best_prec2) + 'Current: [{:.3f}]\t'.format(prec2))
	logger.info('==> Forget Accuracies\tPrevious: [{:.3f}]\t'.format(best_prec3) + 'Current: [{:.3f}]\t'.format(prec3))
	
	
	console_logger.debug("==> unlearning_teacher")
	prec1 = test(loader=valid_dl, model=unlearning_teacher, criterion=criterion, class_mask=class_mask, logger=logger, epoch=epoch)
	prec2 = test(loader=retain_valid_dl, model=unlearning_teacher, criterion=criterion, class_mask=class_mask, logger=logger, epoch=epoch)
	prec3 = test(loader=forget_valid_dl, model=unlearning_teacher, criterion=criterion, class_mask=class_mask, logger=logger, epoch=epoch)

	logger.info('==> Total Accuracies\tPrevious: [{:.3f}]\t'.format(best_prec1) + 'Current: [{:.3f}]\t'.format(prec1))
	logger.info('==> Retain Accuracies\tPrevious: [{:.3f}]\t'.format(best_prec2) + 'Current: [{:.3f}]\t'.format(prec2))
	logger.info('==> Forget Accuracies\tPrevious: [{:.3f}]\t'.format(best_prec3) + 'Current: [{:.3f}]\t'.format(prec3))

	console_logger.info("==> Started adding 9 back")
	for epoch in range(num_passes):
		# Handle lr scheduling
		if epoch <= 0: # Warm start of 1 epoch
			for param_group in optimizer.param_groups:
				param_group['lr'] = opt.maxlr * 0.1
		elif epoch == 1: # Then set to maxlr
			for param_group in optimizer.param_groups:
				param_group['lr'] = opt.maxlr
		else: # Aand go!
			scheduler.step()

		# Train and test loop
		logger.info("==> Starting pass number: "+str(epoch)+", Learning rate: " + str(optimizer.param_groups[0]['lr']))
		model, optimizer = train(opt=opt, loader=train_dl, model=student_model, criterion=criterion, optimizer=optimizer, epoch=epoch, logger=logger)
		prec1 = test(loader=valid_dl, model=student_model, criterion=criterion, class_mask=class_mask, logger=logger, epoch=epoch)
		prec2 = test(loader=retain_valid_dl, model=student_model, criterion=criterion, class_mask=class_mask, logger=logger, epoch=epoch)
		prec3 = test(loader=forget_valid_dl, model=student_model, criterion=criterion, class_mask=class_mask, logger=logger, epoch=epoch)

		# Log performance
		logger.info('==> Current total accuracy: [{:.3f}]\t'.format(prec1))
		logger.info('==> Current retian accuracy: [{:.3f}]\t'.format(prec2))
		logger.info('==> Current forget accuracy: [{:.3f}]\t'.format(prec3))
		if prec1 > best_prec1:
			logger.info('==> Total Accuracies\tPrevious: [{:.3f}]\t'.format(best_prec1) + 'Current: [{:.3f}]\t'.format(prec1))
			best_prec1 = float(prec1)
		if prec2 > best_prec2:
			logger.info('==> Retain Accuracies\tPrevious: [{:.3f}]\t'.format(best_prec2) + 'Current: [{:.3f}]\t'.format(prec2))
			best_prec3 = float(prec2)
		if prec3 > best_prec3:
			logger.info('==> Forget Accuracies\tPrevious: [{:.3f}]\t'.format(best_prec3) + 'Current: [{:.3f}]\t'.format(prec3))
			best_prec4 = float(prec3)

	logger.info('==> Training completed!\n\tTotal Acc: [{0:.3f}]\n\Retian Acc: [{1:.3f}]\n\Forget Acc: [{2:.3f}]'.format(best_prec1, best_prec2, best_prec3))
	
	# Log performance
	logger.info('==> Current total accuracy: [{:.3f}]\t'.format(prec1))
	logger.info('==> Current retian accuracy: [{:.3f}]\t'.format(prec2))
	logger.info('==> Current forget accuracy: [{:.3f}]\t'.format(prec3))
	if prec1 > best_prec1:
		logger.info('==> Total Accuracies\tPrevious: [{:.3f}]\t'.format(best_prec1) + 'Current: [{:.3f}]\t'.format(prec1))
		best_prec1 = float(prec1)
	if prec2 > best_prec2:
		logger.info('==> Retain Accuracies\tPrevious: [{:.3f}]\t'.format(best_prec2) + 'Current: [{:.3f}]\t'.format(prec2))
		best_prec3 = float(prec2)
	if prec3 > best_prec3:
		logger.info('==> Forget Accuracies\tPrevious: [{:.3f}]\t'.format(best_prec3) + 'Current: [{:.3f}]\t'.format(prec3))
		best_prec4 = float(prec3)

	console_logger.debug("==> Completed!")