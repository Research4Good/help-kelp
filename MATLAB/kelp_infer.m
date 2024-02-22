% cd /project/def-lisat-ab/lisat/opensource/kelp/;  run('kelp_gpu0.m')


if 0
    envCfg = coder.gpuEnvConfig('host');
    envCfg.DeepLibTarget = 'cudnn';
    envCfg.DeepCodegen = 1;
    envCfg.Quiet = 1;
    coder.checkGpuInstall(envCfg);
end

if exist('worker') == 0
    spmd
      worker.index = spmdIndex;
      worker.name = system('hostname');
      worker.gpuCount = gpuDeviceCount;
      try
        worker.gpuInfo = gpuDevice;
      catch
        worker.gpuInfo = [];
      end
      worker
    end
end


try        
    checkpointPath = sprintf('%s/gpu_res176_trn0.5_NE%d_BS%d', pwd, encoderDepth, BS);
catch
end





EN=3;
checkpointPath = './gpu_res176_trn0.5_encoder3' % 'net_checkpoint__352__2024_02_21__17_09_21.mat';
checkpointPath = './gpu_res176_20ep';

if 1    
    classNames = ["nokelp", "kelp"];
    pixelLabelIDs = [0, 1];
    if EN==5
        imageSize = [352 352 3];  
    else 
        imageSize = [176 176 3];  
    end    
    encoderDepth = EN;        
    numClasses = 2;    
    lgraph = unetLayers(imageSize,numClasses,'EncoderDepth',encoderDepth)
        
    trn_img_path = './train_features.tar_MLIC14m/train_satellite';
    trn_lab_path = './train_labels.tar_l8u2RP0/train_kelp';
    tst_img_path = './test_features.tar_cVzb1K0/test_satellite/';
            
    if imageSize(1)==352
        trn_lab_dic = pixelLabelDatastore( trn_lab_path, classNames, pixelLabelIDs, ReadFcn=@read_lab_352 );
        trn_img_dic = imageDatastore(trn_img_path, ReadFcn=@read_img_352 );    
        tst_img_dic = imageDatastore(tst_img_path, ReadFcn=@read_img_352 );
    else
        
        if checkpointPath == './gpu_res176_20ep'
            disp('reading old way')
            trn_lab_dic = pixelLabelDatastore( trn_lab_path, classNames, pixelLabelIDs, ReadFcn=@read_lab_0 );
            trn_img_dic = imageDatastore(trn_img_path, ReadFcn=@read_img_0 );    
            tst_img_dic = imageDatastore(tst_img_path, ReadFcn=@read_img_0 );
        else
            trn_lab_dic = pixelLabelDatastore( trn_lab_path, classNames, pixelLabelIDs, ReadFcn=@read_lab);
            trn_img_dic = imageDatastore(trn_img_path, ReadFcn=@read_img );    
            tst_img_dic = imageDatastore(tst_img_path, ReadFcn=@read_img );
        end
    end
    disp( numel(tst_img_dic) )
    
    n = numel(trn_img_dic.Files);
    trn_img = subset(trn_img_dic, 1:2:n);
    trn_lab = subset(trn_lab_dic, 1:2:n);
    val_img = subset(trn_img_dic, 2:2:n);
    val_lab = subset(trn_lab_dic, 2:2:n);
end




mkdir(checkpointPath )
disp( checkpointPath )



if INFER
    files = dir(checkpointPath); 
    files = files( ~[files.isdir] );  
    [~,idx]=sort([files.datenum],'descend'); 
    disp('Load model weights...')
    wt_file = sprintf('%s/%s',checkpointPath, files(idx(1)).name)         
    disp(wt_file)
    tic
    load( wt_file, 'net')
    toc
else

    % https://www.mathworks.com/help/vision/ref/unetlayers.html

    augmenter = imageDataAugmenter( ...
      'RandRotation',[0 360], ...   %%%%Degrees
      'RandXTranslation', [-10 10],...  %%Pixels
      'RandYTranslation', [10 10] );    %%%Pixels

    if 1
        ds     = combine(trn_img, trn_lab);
        ds_val = combine(val_img, val_lab);
    else
        ds     = augmentedImageDatastore(imageSize, trn_img, trn_lab, 'DataAugmentation', augmenter);
        ds_val = augmentedImageDatastore(imageSize, val_img, val_lab, 'DataAugmentation', augmenter);
    end

    % useGPU', 'yes',  ...     'MiniBatchSize', 64, 'Shuffle','every-epoch', ... 
    % "sgdm" | "rmsprop" | "adam" | "lbfgs"
    options = trainingOptions('sgdm', ...
        'InitialLearnRate',1e-3, ...
        'LearnRateSchedule','piecewise', ...
        'LearnRateDropFactor',0.85, ...
        'LearnRateDropPeriod', 3, ... 
        'MiniBatchSize', BS, ...
        'MaxEpochs', 5, ...
        'L2Regularization', 0.0005, ... 
        'ValidationData', ds_val, ...     
        'ValidationFrequency', 4, ...
        'ExecutionEnvironment', 'gpu', ...
        'CheckpointPath', checkpointPath, ...
        'OutputNetwork', 'best-validation-loss', ...
        'VerboseFrequency', 1);



    disp( '\nTraining begins...\n' )        
    tic    
    net = trainNetwork(ds, lgraph, options)
    toc    
end    

disp( '\nPredicting...\n' )        
tic
Y=predict(net, tst_img_dic);
toc

disp( 'Saving...' )        
save( sprintf('%s_tst_yps.mat', checkpointPath), 'Y');
out_dir = sprintf('%s_submission/', checkpointPath)
mkdir( out_dir )

n_tsts= numel(tst_img_dic.Files);   
areas = zeros( n_tsts,1 );

Y2=imresize( Y, [352 352]);        

tic
run('kelp_postprocess.m')    
toc


  

function t = read_lab_0(filename)    
    t = imread(filename);    
    t = padarray( t, [1,1,0] );
    t = t(1:2:end,1:2:end);
    t = uint8(t);
end

function t = read_img_0(filename)    
    t = imread(filename);
    t( t==-32768 ) = 1;
    t = single(t);
    t = cat(3, t(:,:,1), t(:,:,2), rgb2gray( t(:,:,3:5)) );
    t = padarray( t, [1,1,0] );
    t = t(1:2:end,1:2:end,:)*1.;
    t = t/65536;
end




function t = read_lab(filename)    
    t = imread(filename);    
    t = padarray( t, [1,1,0] );
    t = t(1:2:end,1:2:end);
    t = uint8(t);
    % disp(  sum(t(t==1)) )
end

function t = read_img(filename)    
    t = imread(filename);
    
    % remove NaN
    t( t==-32768 ) = 0;
    
    % convert dtype
    t = single(t); 
    
    % (1 - np.uint8( img[:,:,ch]>k ) )*img[:,:,ch]  # Python     
        
    k=10000; % determined threshold via visual inspect on random subset
    band2 = t(:,:,2);
    
    % focus learning on kelp-probable areas
    mask = (1- single(band2 >k) );

    % pixel-wise multiplication in Matlab
    
    band1 = mask.*t(:,:,1);   
    band2 = mask.*t(:,:,2);
    
    t = cat(3, band1/k, band2/k, rgb2gray( 255*t(:,:,3:5)/65536 ) );
    t = padarray( t, [1,1,0] );
    
    t = t(1:2:end,1:2:end,:)*1.;    
    
end


function t = read_lab_352(filename)    
    t = imread(filename);    
    t = padarray( t, [1,1,0] );    
    t = uint8(t);
    % disp(  sum(t(t==1)) )
end

function t = read_img_352(filename)    
    t = imread(filename);
    
    % remove NaN
    t( t==-32768 ) = 0;
    
    % convert dtype
    t = single(t); 
    
    % (1 - np.uint8( img[:,:,ch]>k ) )*img[:,:,ch]  # Python     
        
    k=10000; % determined threshold via visual inspect on random subset
    band2 = t(:,:,2);
    
    % focus learning on kelp-probable areas
    mask = (1- single(band2 >k) );

    % pixel-wise multiplication in Matlab
    
    band1 = mask.*t(:,:,1);   
    band2 = mask.*t(:,:,2);
    
    t = cat(3, band1/k, band2/k, rgb2gray( 255*t(:,:,3:5)/65536 ) );
    t = padarray( t, [1,1,0] )*1;        
    
end
