function data_trva()
    seed = 123456789;
    rng(seed);

    name = 'data';
    
    a = 1.0;
    gamma = 1.0;

    tx = linspace(-2, 2, 4);
    ty = linspace(-2, 2, 4);
    [tx, ty] = meshgrid(tx,ty);
    inits_tr = [reshape(tx,[],1), reshape(ty,[],1)];    
    tx = linspace(-1.5, 1.5, 4);
    ty = linspace(-2, 2, 4);
    [tx, ty] = meshgrid(tx,ty);
    inits_va = [reshape(tx,[],1), reshape(ty,[],1)];
    clear t tx ty;

    len_epi = 80; tmax = 4.0;
    inits = [inits_tr; inits_va];
    num_epi = size(inits,1);
    tr = 1:size(inits_tr,1);
    va = tr(end)+1:tr(end)+size(inits_va,1);

    T_all = zeros(num_epi, len_epi);
    Y_all = zeros(num_epi, len_epi, 2);
    dotY_all = zeros(num_epi, len_epi, 2);
    for i=1:num_epi
        [T_, Y_, dotY_] = linear(len_epi, inits(i,:), tmax, a, gamma);
        T_all(i,:) = T_;
        Y_all(i,:,:) = Y_;
        dotY_all(i,:,:) = dotY_;
    end

    noisestd = 0;

    Y_all = Y_all + randn(size(Y_all))*noisestd;
    dotY_all = dotY_all + randn(size(dotY_all))*noisestd;

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
