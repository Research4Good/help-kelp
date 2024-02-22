tic
for i = 1:n_tsts
    [p,n,e]=fileparts(tst_img_dic.Files{i});
    a=split(n,'_');
    a=string(a(1,1));    
    im2 = Y2(:,:,2,i)>Y2(:,:,1,i);             
    ar = sum(im2(:));            
    areas(i) = ar;       
    im2=im2(2:end-1,2:end-1);  % chop out                   
    
    if 1
        src = imread( sprintf( '/project/def-lisat-ab/lisat/opensource/kelp/test_features.tar_cVzb1K0/test_satellite/%s.tif', n) );
        src(src<0) = 0;
        src = src(:,:,2);
        mask = single( src  < 10000 );
        fname = sprintf('%s/%s_kelp.tif', out_dir, a);
        imwrite( mat2gray(im2.*mask), fname );        
        disp( sprintf( '%s %d', a, ar ) );        
    end
end    
toc
