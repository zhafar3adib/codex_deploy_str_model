with stockawal as
(
select
artikel,sub_kategori,
sum(stock) stock,
min(s.tanggal) tanggal,
date_sub(min(s.tanggal),interval 90 day) tanggal_awal,
-- date_add(min(s.tanggal),interval 30 day) tanggal_akhir,
-- date_add(min(s.tanggal),interval 60 day) tanggal_akhir2
from `klamby-469003.sales.stockawal` s
left join `klamby-469003.sales.master_sku_gabungan` ms on s.sku = ms.sku
where keterangan != 'Restock' and ms.artikel is not null and ms.artikel not like '%Minor%' and ms.artikel not like '%Major%'
group by 1,2
),
sold as
(
select date_trunc(created_at, day) tanggal, ms.artikel, sum(quantity) quantity
from (select * from `klamby-469003.sales.daily` union all select * from `klamby-469003.sales.monthly`) n
left join `klamby-469003.sales.master_sku_gabungan` ms on n.sku = ms.sku
where ms.artikel is not null and ms.artikel not like '%Minor%' and ms.artikel not like '%Major%'
and (status_id = 101 or status_id = 401)
group by 1,2
),
product_attribute as
(
SELECT
*,
row_number() over(partition by artikel) rn
FROM `klamby-469003.sales.product_attribute`
qualify rn = 1
)
select
s.artikel, s.tanggal,s.sub_kategori, stock,
upper(p.cutting) cutting, upper(p.design_density) design_density, upper(p.design_border) design_border,upper(p.siluet) siluet,
upper(p.sleeve) sleeve,upper(p.collar) collar,upper(p.length_hem) length_hem,upper(p.color_theme) color_theme,upper(p.closure) closure,upper(p.embellishment) embellishment,
sum(so.quantity) quantity,
sum(case when so.tanggal >= s.tanggal_awal and so.tanggal <= date_add(s.tanggal, interval 30 day) then so.quantity else 0 end) as quantity30,
sum(case when so.tanggal >= s.tanggal_awal and so.tanggal <= date_add(s.tanggal, interval 60 day) then so.quantity else 0 end) as quantity60,
sum(case when so.tanggal >= s.tanggal_awal and so.tanggal <= date_add(s.tanggal, interval 90 day) then so.quantity else 0 end) as quantity90
from stockawal s
left join sold so on s.artikel = so.artikel and so.tanggal >= tanggal_awal --and so.tanggal <= tanggal_akhir
--left join sold so2 on s.artikel = so2.artikel and so.tanggal >= tanggal_awal and so2.tanggal <= tanggal_akhir2
--where s.artikel = 'Elaya Sock'
left join product_attribute p on UPPER(s.artikel) = p.artikel
where p.artikel is not null AND DATE(s.tanggal) <= DATE_SUB(@as_of, INTERVAL 90 DAY)
group by 1,2,3,4,5,6,7,8,9,10,11,12,13,14
