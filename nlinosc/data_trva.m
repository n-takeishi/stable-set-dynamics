function data_trva()
    seed = 1234;
    rng(seed);

    name = 'data';
    
    mu = 2;

    tx = linspace(-2.5, 2.5, 20);
    ty = linspace(-4.5, 4.5, 20);
    [tx, ty] = meshgrid(tx,ty);
    inits_tr = [reshape(tx,[],1), reshape(ty,[],1)];
    tx = linspace(-2, 2, 15);
    ty = linspace(-4, 4, 15);
    [tx, ty] = meshgrid(tx,ty);
    inits_va = [reshape(tx,[],1), reshape(ty,[],1)];
    inits = [inits_tr; inits_va];
    num_epi_all = size(inits,1);
    tr = 1:size(inits_tr,1); va = tr(end)+1:tr(end)+size(inits_va,1);
    clear tx ty inits_tr inits_va;
    len_epi = 1; tmax = 0.02;

    T_all = zeros(num_epi_all, len_epi);
    Y_all = zeros(num_epi_all, len_epi, 2);
    dotY_all = zeros(num_epi_all, len_epi, 2);
    for i=1:num_epi_all
        if len_epi>2
            [T_, Y_, dotY_] = vdp(len_epi, inits(i,:), mu, tmax);
        else
            T_=0.0;
            Y_=inits(i,:);
            dotY_= [Y_(2); mu*(1-Y_(1)^2)*Y_(2)-Y_(1)];
        end
        T_all(i,:) = T_;
        Y_all(i,:,:) = Y_;
        dotY_all(i,:,:) = dotY_;
    end

    noisestd = 0;

    Y_all = Y_all + randn(size(Y_all))*noisestd;
    dotY_all = dotY_all + randn(size(dotY_all))*noisestd;

    [~, lc, ~] = vdp(1000, [0.369,3.298], 2, 20.0);

    % plot
    if true
        figure;
        hold on;
        for i=tr
            plot(Y_all(i,:,1), Y_all(i,:,2), 'b');
            plot(inits(i,1), inits(i,2), 'bo')
        end
        for i=va
            plot(Y_all(i,:,1), Y_all(i,:,2), 'r');
            plot(inits(i,1), inits(i,2), 'ro')
        end
        plot(lc(:,1), lc(:,2), 'k--');
        hold off;
    end
    
    % save
    if true
        T = T_all(tr,:);
        Y = Y_all(tr,:,:);
        dotY = dotY_all(tr,:,:);
        save(sprintf('./%s_tr.mat',name), 'T', 'Y', 'dotY');

        T = T_all(va,:);
        Y = Y_all(va,:,:);
        dotY = dotY_all(va,:,:);
        save(sprintf('./%s_va.mat',name), 'T', 'Y', 'dotY');

        clear T Y dotY T_all Y_all dotY_all;
        save(sprintf('./%s_info.mat',name));
        
        fprintf('data saved\n');
    end
end
