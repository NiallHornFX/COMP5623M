# ======= Single Batch Training =======
test_images, test_labels = iter(train_loader).next()  
test_images, test_labels = test_images.cuda(device), test_labels.cuda(device)

for epoch in range(n_epochs):
    opt_fn.zero_grad()
    
    # Eval network (forward, compute loss, backprop)
    outputs = model(test_images) 
    loss = loss_fn(outputs, test_labels)
    loss.backward()
    opt_fn.step()
        
    # Store Training Loss
    train_loss = loss 

    # Run over Validation set
    running_valid_loss = 0.0
    n = 0
    with torch.no_grad():
        for i, data in enumerate(valid_loader, 0):
            valid_images, valid_labels = data[0].to(device), data[1].to(device)
            valid_out = model(valid_images)
            valid_loss = loss_fn(valid_out, valid_labels)
            running_valid_loss += loss.item()
            n += 1
    # Calc Total Valid Loss
    running_valid_loss /= n
    
    print(f"epoch: {epoch} training_loss: {loss: .3f}")


    # ====== Validation (All Batches) 
    running_valid_loss = 0.0
    n = 0
    # Run over Validation set
    with torch.no_grad():
        for i, data in enumerate(valid_loader, 0):
            # Pass validation batch Images and Labels onto device
            valid_images, valid_labels = data[0].to(device), data[1].to(device)
            
            #valid_images, valid_labels = data
            
            # Eval model for validation images and compute loss
            valid_out = model(valid_images)
            valid_loss = loss_fn(valid_out, valid_labels)
            running_valid_loss += valid_loss.item()
            n += 1
            
    # Calc total Validation Loss
    running_valid_loss /= n
    
    # Store Output per epoch
    store[0][epoch] = train_loss
    store[1][epoch] = running_valid_loss
    print(f"epoch: {epoch} training_loss: {loss: .5f} validation_loss: {running_valid_loss: .5f}")     
    # Print Output per epoch (training vs valid loss) every nth epoch
    #if epoch % 10 == 0:
        #print(f"epoch: {epoch} training_loss: {loss: .5f} validation_loss: {running_valid_loss: .5f}")   