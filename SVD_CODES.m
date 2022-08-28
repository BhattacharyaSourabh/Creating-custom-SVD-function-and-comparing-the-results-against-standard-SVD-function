%Name: Sourabh Bhattacharya
%Roll No. : 21EE64R18
g = [ 255   255   255   255   255   255   255   255;
      255   255   255   100   100   100   255   255;
      255   255   100   150   150   150   100   255;
      255   255   100   150   200   150   100   255;
      255   255   100   150   150   150   100   255;
      255   255   255   100   100   100   255   255;
      255   255   255   255   50    255   255   255;
      50    50    50    50    255   255   255   255; ];

f=[];  new = [];
  
figure;
subplot(1,2,1);
imshow(g,[])
title("Original Image")

[ww,xx,yy] = svd(g);
recon_g = ww*xx*yy';
%figure;
subplot(1,2,2)
imshow(recon_g,[])
title("Standard Function")
p= zeros(8,8);
p(1,1) = xx(1,1);
recon_1g  = ww*p*yy';
figure;
subplot(1,5,1)
imshow(recon_1g,[])
p(2,2)  = xx(2,2);
j = ww*p*yy';
title("k="+1)
subplot(1,5,2)
imshow(j,[])
p(3,3)=xx(3,3);
j2=ww*p*yy';
title("k="+2)
subplot(1,5,3);
imshow(j2,[])
p(4,4)=xx(4,4);
j3 = ww*p*yy';
title("k="+3)
subplot(1,5,4);
imshow(j3,[]);
p(5,5) = xx(5,5);
j4 = ww*p*yy';
title("k="+4)
subplot(1,5,5);
imshow(j4,[])
title("k="+5)


f=[f norm(g-recon_1g)];
f=[f norm(g-j)];
f=[f norm(g-j2)];
f=[f norm(g-j3)];
f=[f norm(g-j4)];


%figure;
%plot(x,f);
%title("Original vs Modified")




graph=[];
for k=1:5
[ww,xx,yy,zz] = self_SVD(g,k,0.001);
new = [new zz];
xx = diag(xx);
dev_g = ww*xx*yy;
graph=[graph norm(g-dev_g)];
figure;
subplot (2,1,1);
imshow(g,[]);
title("For K="+k)
subplot (2,1,2);
imshow(dev_g,[]);
title("For K="+k)
end

%x = 1:k;
%figure;
%plot(x,graph);
%title("Original vs SVD from SCRATCH")

figure;
for j = 1:k
    [ww,xx,yy,zz] = self_SVD(g,k,0.001);
    hold on
    plot(zz);
    title("Convergence with different values of K........")
end

function [a,b,c,d] = self_SVD(mat,iter,eps_chk)
    [q,r] = size(mat);
    if iter>min(q,r)
        iter=min(q,r);
    end

    a = zeros(q,r);
    c = zeros(q,r);
    b = zeros(iter,1);
    function present_vec = Dom_eig(original,eps_tol)
        [~,sec]=size(original);
        new = randn(sec,1);
        stored = new/norm(new);
        present_vec = stored;
        nxt_mat = zeros(size(stored));
        covariance = original'*original;
        d=[];
        count=0;
        while 1
            count=count+1;
            nxt_mat = present_vec;
            present_vec = covariance*nxt_mat;
            present_vec = present_vec/norm(present_vec);
            d = [d 1-abs(present_vec'*nxt_mat)];
            if abs(present_vec'*nxt_mat)>(1-eps_tol)
                break
            end
        end
    end
    q_var = zeros(q,iter);
    r_var = zeros(r,iter);
    for i=1:iter
        modified_mat = mat;
        for j=1:i
            modified_mat = modified_mat - b(j,1)*q_var(:,j)*r_var(:,j)';
        end
        r_var(:,i) = Dom_eig(modified_mat,eps_chk);
        u_sig = mat*r_var(:,i);
        b(i) = norm(u_sig);
        q_var(:,i) = u_sig/b(i);
        if i>=2
            if b(i)>b(i-1)
                b(i)=0;
            end
        end
    end
    a = q_var;
    c = r_var';
end


  
    